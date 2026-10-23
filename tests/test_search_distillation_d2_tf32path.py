"""The full-float32 setting's policy-head and full-gradient-path validation (`scripts/GPU/alphazero/
search_distillation_d2_tf32path.py`): its refusals, the REAL setting check in fresh processes, the stop before each
later phase, retention of every earlier result, and one end-to-end run of the public entry in the REAL state (the gate
open by monkeypatch, the pinned card, code and evidence, entry 26's real saved fixtures, the real corpus through the
training-only reader and the real incumbent, all read-only). Both device labels run on the CPU here ("gpu" is
remapped), so every honest comparison agrees to the bit and every disagreement is an injected fault. The phase tests
use the earlier diagnostics' fixture rig (a small network, a 4-update reference); their "U21" is the state after 3
updates.

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture remaps the devices and
redirects the evidence for every in-process test. The checks of the REAL setting and of MLX already being loaded need a
fresh process; those subprocesses redirect the evidence and replace `conclude` with a stub, so even a mutant that
removes every check before it cannot reach a real run."""
import json
import os
import pathlib
import signal
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from scripts.GPU.alphazero import search_distillation_d2_gradcheck as K  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_measure as M  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_tf32path as E  # noqa: E402
from scripts.GPU.alphazero import search_distillation_d2_tf32path_authorization as GATE  # noqa: E402
from scripts.GPU.alphazero import training_replay_capture as C  # noqa: E402
from tests.test_search_distillation_d2_gradcheck import kit  # noqa: E402,F401
from tests.test_search_distillation_d2_gradrepeat import done, rig  # noqa: E402,F401
from tests.test_search_distillation_d2_gradtrace import ref, reproduced  # noqa: E402,F401
from tests.test_search_distillation_d2_measure import _net  # noqa: E402
from tests.test_search_distillation_d2_train import corpus  # noqa: E402,F401

PHASE_FILES = ["reference/repeated_indices.json", "reference/batch_members.json", "reference/padding.json",
               "reference/weighted_additivity.json", "reference/real_S0.json", "reference/real_S21.json"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(K, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})
    monkeypatch.setattr(E, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(E, "_mlx_loaded", lambda: False)          # this process has MLX; the subprocesses test it
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED", True)
    return isolated


def _checkpoint() -> pathlib.Path:
    """The real incumbent, from the main checkout (the parent of the shared git directory), as the wrapper finds it."""
    common = subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=E.ROOT,
                                     text=True).strip()
    return pathlib.Path(common).parent / M.CHECKPOINT


def _spec(kit):                                                         # noqa: F811
    shared = ("corpus_sha256", "teacher_digest", "reference_sha256", "reference_student_digest",
              "training_ids_sha256", "training_games", "held_out_games", "training_records", "passes", "steps",
              "batch_size", "learning_rate")
    return {**E.FROZEN, **{k: kit["spec"][k] for k in shared}, "batches": [0, 3], "prefix_updates": 3, "repeats": 2}


def _seams(kit, **over):                                                # noqa: F811
    rig = kit["rig"]
    base = dict(read_corpus_json=lambda: rig["corpus"]["raw"], read_reference=lambda: rig["ref"]["raw"],
                checkpoint_sha1=lambda: M.CHECKPOINT_SHA1, load_network=_net,
                loaded_modules=lambda: {"fixture": "not a closure"}, read_repo=lambda rel: (E.ROOT / rel).read_bytes(),
                read_fixture=lambda path: pathlib.Path(path).read_bytes())
    base.update(over)
    return E.Seams(**base)


def _go(kit, tmp_path, spec=None, **over):                              # noqa: F811
    return E.conclude(spec or _spec(kit), _seams(kit, **over), tmp_path / "ev")


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def _on_gpu_label(make, fault):
    """A network factory whose networks apply `fault` to forward_padded's outputs when built under the "gpu" label."""
    def load():
        net = make()
        if K.CURRENT["device"] == "gpu":
            real = net.forward_padded
            net.forward_padded = lambda *a, **k: fault(*real(*a, **k))
        return net
    return load


# ─────────────────────────────── refusals ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED is False
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err and not (isolated / "evidence").exists()


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_DEVICES_OR_EVIDENCE(isolated):
    assert K.DEVICES == {"cpu": mx.cpu, "gpu": mx.cpu} and E.on is K.on
    assert E.EVIDENCE_DIR == str(isolated / "evidence")


def test_ANYTHING_BUT_run_WITH_A_CHECKPOINT_IS_USAGE():
    assert E.main([]) == 2 and E.main(["run"]) == 2 and E.main(["run", "--x", "/y"]) == 2


def test_THE_PINS_ARE_THE_FILES_ON_DISK_and_the_tolerances_are_entry_26_s():
    assert M._sha((E.ROOT / E.CARD).read_bytes()) == E.CARD_SHA256
    for rel, sha in E.PINNED.items():
        assert M._sha((E.ROOT / rel).read_bytes()) == sha, rel
    assert pathlib.Path(K.__file__).resolve() == E.ROOT / E.CODE["gradcheck"]
    assert {k: E.FROZEN[k] for k in ("cross_rel_tol", "cross_abs_tol", "fd_eps", "fd_rel_tol", "fd_mlx_rel_tol",
                                     "fd_seed", "devices", "repeats")} == \
        {k: K.FROZEN[k] for k in ("cross_rel_tol", "cross_abs_tol", "fd_eps", "fd_rel_tol", "fd_mlx_rel_tol",
                                  "fd_seed", "devices", "repeats")} == \
        {"cross_rel_tol": 1e-3, "cross_abs_tol": 1e-6, "fd_eps": 1e-5, "fd_rel_tol": 1e-6, "fd_mlx_rel_tol": 1e-3,
         "fd_seed": 0, "devices": ["cpu", "gpu"], "repeats": 3}
    assert (E.FROZEN["batches"], E.FROZEN["prefix_updates"]) == ([0, 21], 21)
    assert E.FROZEN["policy_head_cases"][:4] == E.FROZEN["directional_required"]
    assert not E.FIXTURE_DIR.is_relative_to(E.ROOT)


@pytest.mark.parametrize("pin", [E.CODE["gradcheck"], f"{E.ENTRY26}/fixtures.jsonl", f"{E.ENTRY26}/full.jsonl"])
def test_A_FILE_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    monkeypatch.setitem(E.PINNED, pin, "0" * 64)
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "pinned" in capsys.readouterr().err and not (opened / "evidence").exists()


def test_AN_UNPINNED_CARD_REFUSES(opened, monkeypatch, capsys):
    monkeypatch.setattr(E, "CARD_SHA256", "1" * 64)
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "not the pinned card" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "1", "", " 0", "false"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "MLX_ENABLE_TF32" in capsys.readouterr().err and not (opened / "evidence").exists()


def test_MLX_ALREADY_LOADED_REFUSES_even_with_the_gate_open(opened, monkeypatch, capsys):
    monkeypatch.setattr(E, "_mlx_loaded", lambda: True)
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "imported" in capsys.readouterr().err and not (opened / "evidence").exists()


def test_AN_EXISTING_EVIDENCE_DIRECTORY_A_RELATIVE_CHECKPOINT_OR_NO_FIXTURES_REFUSE(opened, monkeypatch, capsys):
    (opened / "evidence").mkdir()
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "one-shot" in capsys.readouterr().err and list((opened / "evidence").iterdir()) == []
    (opened / "evidence").rmdir()
    assert E.main(["run", "--checkpoint", "checkpoints/x.safetensors"]) == E.EXIT_REFUSED
    assert "absolute path" in capsys.readouterr().err
    monkeypatch.setattr(E, "FIXTURE_DIR", opened / "nowhere")
    assert E.main(["run", "--checkpoint", str(_checkpoint())]) == E.EXIT_REFUSED
    assert "not a directory" in capsys.readouterr().err and not (opened / "evidence").exists()


# ─────────────────────────────── the REAL setting check, in fresh processes ───────────────────────────────

PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import search_distillation_d2_tf32path as E
from scripts.GPU.alphazero import search_distillation_d2_tf32path_authorization as GATE
print("MLX_AFTER_IMPORT", E._mlx_loaded(), flush=True)
E.EVIDENCE_DIR = str(pathlib.Path(tempfile.mkdtemp()) / "evidence")
def stub(spec, seams, out, head=None):     # never a real run, whatever a mutant does to the checks above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB", "head": head}
E.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(E.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    argv = repr(["run", "--checkpoint", str(_checkpoint())])
    return subprocess.run([sys.executable, "-c", PRELUDE + body + f"\nsys.exit(E.main({argv}))\n"], cwd=E.ROOT,
                          env=env, capture_output=True, text=True, timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the three phases, on the rig ───────────────────────────────

@pytest.fixture(scope="module")
def clean(kit, tmp_path_factory):                                       # noqa: F811
    mp = pytest.MonkeyPatch()
    mp.setattr(K, "DEVICES", {"cpu": mx.cpu, "gpu": mx.cpu})
    mp.setenv("MLX_ENABLE_TF32", "0")
    d = tmp_path_factory.mktemp("tf32path")
    try:
        code, rec = E.conclude(_spec(kit), _seams(kit), d / "ev", head="f" * 40)
    finally:
        mp.undo()
    return code, rec, d


def test_A_CLEAN_RUN_PASSES_ALL_THREE_PHASES_and_writes_each_result_and_no_array(clean):
    code, rec, d = clean
    ev = d / "ev"
    assert code == E.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert json.loads((ev / "record.json").read_text()) == json.loads(json.dumps(rec, sort_keys=True))
    assert rec["verdict"] == "PASS" and rec["failed_phase"] is None and rec["not_run"] == []
    assert {k: v["status"] for k, v in rec["phases"].items()} == dict.fromkeys(E.PHASES, "PASS")
    assert rec["setting"] == rec["setting_at_end"] == {"MLX_ENABLE_TF32": "0"}
    ph = rec["phases"]["policy_head"]
    assert list(ph["cases"]) == E.FROZEN["policy_head_cases"]
    assert {c: v["directional_skipped"] for c, v in ph["cases"].items()} == {
        "repeated_indices": False, "batch_members": False, "padding": False, "weighted_additivity": False,
        "real_S0": True, "real_S21": True}                              # as in entry 26
    assert rec["entry26"]["policy_head"]["cpu_all_equal"] is True, rec["entry26"]   # prediction 1's, on the CPU
    for rel in PHASE_FILES + ["full/S0_b0.json", "full/S0_b3.json", "full/U21_b0.json", "full/U21_b3.json"]:
        assert (ev / rel).exists(), rel
    full = _lines(ev / "full.jsonl")
    assert [(x["block"], x["device"], x["repeat"]) for x in full] == [
        (f"{s}_b{b}", dev, r) for s in ("S0", "U21") for b in (0, 3) for dev in ("cpu", "gpu") for r in (0, 1)]
    blk = json.loads((ev / "full" / "U21_b3.json").read_text())["devices"]["gpu"]
    assert blk["failures"] == [] and len(blk["grads_vs_reference"][0]) == len(blk["additivity"]) > 0
    assert all(v["bitwise"] for v in blk["logits_legal_vs_reference"]) and all(blk["logits_padding_equal"])
    recon = _lines(ev / "reconstruction.jsonl")
    assert [x["update"] for x in recon] == [0, 1, 2] and all("stage3" in x for x in recon)
    assert rec["updated_state"]["saved"] is False and rec["updated_state"]["updates"] == 3
    assert rec["held_out_records_read"] == 0 and rec["model_state_saved"] is False and rec["arrays_saved"] is False
    assert not [p for p in d.rglob("*") if p.suffix in (".npz", ".npy", ".safetensors")]


def test_A_POLICY_HEAD_FAULT_ON_THE_GPU_FAILS_PHASE_1_AND_NOTHING_LATER_RUNS(kit, tmp_path, monkeypatch):  # noqa
    real = K.mlx_policy_head

    def head(*a):
        L, g, logits = real(*a)
        if K.CURRENT["device"] == "gpu":
            g = {**g, "fc.weight": g["fc.weight"] * 1.01}
        return L, g, logits

    monkeypatch.setattr(K, "mlx_policy_head", head)
    code, rec = _go(kit, tmp_path, load_network=lambda: pytest.fail("a network was loaded after phase 1 failed"))
    assert code == E.EXIT_COMPLETED and rec["verdict"] == "FAIL" and rec["failed_phase"] == "policy_head"
    assert rec["not_run"] == ["full_model_incumbent", "full_model_updated"] and list(rec["phases"]) == ["policy_head"]
    gpu = rec["phases"]["policy_head"]["gpu_failures"]
    assert {f"{c}: fc.weight vs NumPy" for c in E.FROZEN["policy_head_cases"]} <= set(gpu), gpu
    assert "padding: directional vs central difference" in gpu
    assert not [x for x in gpu if x.endswith(" additivity")]           # a uniform scale keeps the sum additive
    assert rec["phases"]["policy_head"]["cpu_own_failures"] == [] and "teacher_values" not in rec
    ev = tmp_path / "ev"
    assert all((ev / r).exists() for r in PHASE_FILES) and not (ev / "full.jsonl").exists()
    assert list((ev / "full").iterdir()) == [] and not (ev / "reconstruction.jsonl").exists()


def test_A_FULL_MODEL_FAULT_ON_THE_GPU_FAILS_PHASE_2_AND_NO_UPDATE_RUNS(kit, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(K, "reconstruct", lambda *a, **k: pytest.fail("an optimizer update ran after phase 2 failed"))
    load = _on_gpu_label(_net, lambda logits, values, *rest: (logits, values * 1.01, *rest))
    code, rec = _go(kit, tmp_path, load_network=load)
    assert code == E.EXIT_COMPLETED and rec["verdict"] == "FAIL" and rec["failed_phase"] == "full_model_incumbent"
    assert rec["not_run"] == ["full_model_updated"] and rec["phases"]["policy_head"]["status"] == "PASS"
    gpu = rec["phases"]["full_model_incumbent"]["gpu_failures"]
    assert "S0_b0: evaluation 0: values" in gpu and "S0_b3: evaluation 1: values" in gpu, gpu
    assert rec["phases"]["full_model_incumbent"]["cpu_own_failures"] == []
    ev = tmp_path / "ev"
    blk = json.loads((ev / "full" / "S0_b0.json").read_text())["devices"]["gpu"]["values_vs_reference"][0]
    assert blk["within_tol"] is False and blk["n_outside"] > 0 and set(blk["worst"]) == {"index", "got", "reference"}
    assert not (ev / "reconstruction.jsonl").exists() and "updated_state" not in rec
    assert sorted(p.name for p in (ev / "full").iterdir()) == ["S0_b0.json", "S0_b3.json"]


def test_A_FAULT_ONLY_AFTER_THE_PREFIX_FAILS_PHASE_3(kit, tmp_path, monkeypatch):  # noqa: F811
    real, state = K.reconstruct, {"after": False}

    def reconstruct(*a, **k):
        out = real(*a, **k)
        state["after"] = True
        return out

    monkeypatch.setattr(K, "reconstruct", reconstruct)
    gpu_net = _on_gpu_label(_net, lambda logits, values, *rest: (logits * 1.01, values, *rest))
    code, rec = _go(kit, tmp_path, load_network=lambda: gpu_net() if state["after"] else _net())
    assert code == E.EXIT_COMPLETED and rec["verdict"] == "FAIL" and rec["failed_phase"] == "full_model_updated"
    assert rec["phases"]["full_model_incumbent"]["status"] == "PASS" and rec["not_run"] == []
    assert "U21_b0: evaluation 0: legal logits" in rec["phases"]["full_model_updated"]["gpu_failures"]
    assert len(_lines(tmp_path / "ev" / "reconstruction.jsonl")) == 3


def test_A_NON_REPEATABLE_GPU_GRADIENT_FAILS_even_within_the_tolerance(kit, tmp_path):  # noqa: F811
    calls = []

    def fault(logits, values, *rest):
        calls.append(1)
        return logits, values * (1 + 1e-7 * len(calls)), *rest

    code, rec = _go(kit, tmp_path, load_network=_on_gpu_label(_net, fault))
    assert code == E.EXIT_COMPLETED and rec["failed_phase"] == "full_model_incumbent"
    gpu = rec["phases"]["full_model_incumbent"]["gpu_failures"]
    assert "S0_b0: repeat 1: values" in gpu and not [x for x in gpu if "evaluation" in x and "values" in x]


def test_SIGTERM_DURING_THE_PREFIX_IS_A_VOID_and_phases_1_and_2_survive(kit, tmp_path, monkeypatch):  # noqa: F811
    def reconstruct(*a, **k):
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("SIGTERM did not stop the run")

    monkeypatch.setattr(K, "reconstruct", reconstruct)
    previous = signal.getsignal(signal.SIGTERM)
    code, rec = _go(kit, tmp_path)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert code == E.EXIT_VOID and rec["void"]["step"] == "terminated" and "verdict" not in rec
    assert {k: v["status"] for k, v in rec["phases"].items()} == {"policy_head": "PASS",
                                                                  "full_model_incumbent": "PASS"}
    ev = tmp_path / "ev"
    assert json.loads((ev / "record.json").read_text())["status"] == "VOID"
    assert all((ev / r).exists() for r in PHASE_FILES + ["full/S0_b0.json", "full/S0_b3.json"])
    assert len(_lines(ev / "full.jsonl")) == 8


def test_A_NON_FINITE_GPU_RESULT_IS_A_VOID_and_earlier_lines_survive(kit, tmp_path):  # noqa: F811
    load = _on_gpu_label(_net, lambda logits, values, *rest: (logits, values * float("nan"), *rest))
    code, rec = _go(kit, tmp_path, load_network=load)
    assert code == E.EXIT_VOID and rec["void"]["step"] == "non_finite" and "S0_b0" in rec["void"]["detail"]
    assert rec["phases"]["policy_head"]["status"] == "PASS"
    assert [(x["device"], x["repeat"]) for x in _lines(tmp_path / "ev" / "full.jsonl")] == [("cpu", 0), ("cpu", 1)]


@pytest.mark.parametrize("case, step", [("fixture_bytes", "fixtures"), ("fixture_path", "fixtures"),
                                        ("entry26", "entry26"), ("checkpoint", "checkpoint"), ("plan", "plan"),
                                        ("corpus", "corpus"), ("reference", "reference")])
def test_A_BROKEN_INPUT_IS_A_VOID_BEFORE_PHASE_1(kit, tmp_path, monkeypatch, case, step):  # noqa: F811
    spec, over = _spec(kit), {}
    if case == "fixture_bytes":
        over["read_fixture"] = lambda path: b"x" if path.endswith("reference_padding.npz") else \
            pathlib.Path(path).read_bytes()
    elif case == "fixture_path":
        monkeypatch.setattr(E, "FIXTURE_DIR", tmp_path / "elsewhere")
    elif case == "entry26":
        over["read_repo"] = lambda rel: (E.ROOT / rel).read_bytes() + (b" " if rel.endswith("real_S0.json") else b"")
    elif case == "checkpoint":
        over["checkpoint_sha1"] = lambda: "0" * 40
    elif case == "plan":
        spec["batches"] = [0, 2]
    elif case == "corpus":
        over["read_corpus_json"] = lambda: kit["rig"]["corpus"]["raw"] + b" "
    elif case == "reference":
        over["read_reference"] = lambda: kit["rig"]["ref"]["raw"] + b" "
    code, rec = _go(kit, tmp_path, spec, **over)
    assert code == E.EXIT_VOID and rec["void"]["step"] == step, rec.get("void")
    assert rec["phases"] == {} and not (tmp_path / "ev" / "reference").exists()


def test_THE_REFERENCE_MUST_VALIDATE_ITSELF_ON_THE_SYNTHETIC_CASES(kit, tmp_path, monkeypatch):  # noqa: F811
    real = K.directional_check

    def check(p, F, *a, **k):
        out = real(p, F, *a, **k)
        return {"eps": 1e-5, "skipped": True, "reason": "forced"} if np.shape(F)[0] == 3 else out   # padding: B = 3

    monkeypatch.setattr(K, "directional_check", check)
    code, rec = _go(kit, tmp_path)
    assert code == E.EXIT_VOID and rec["void"]["step"] == "reference" and "padding" in rec["void"]["detail"]
    monkeypatch.setattr(K, "directional_check", lambda *a, **k: {**real(*a, **k), "analytic_ok": False})
    code, rec = _go(kit, tmp_path / "again")
    assert code == E.EXIT_VOID and rec["void"]["step"] == "reference"


def test_A_REBUILT_HEAD_MUST_READ_BACK_AS_THE_SAVED_ONE():
    raw = (E.FIXTURE_DIR / "reference_padding.npz").read_bytes()
    a = dict(np.load(__import__("io").BytesIO(raw)))
    ph = E.rebuild_head(a, "padding")
    assert all(np.array_equal(K.policy_head_params(ph)[k], a[f"param.{k}"]) for k in K.PH_KEYS)
    bad = {**a, "param.fc.bias": a["param.fc.bias"] + 1e-12}             # not a float32 value
    with pytest.raises(E.Void, match="rebuilt policy head"):
        E.rebuild_head(bad, "padding")


def test_JUDGE_BLOCK_names_every_failing_comparison_and_the_padding():
    ok = {"bitwise": True, "within_tol": True}
    bad = {"bitwise": False, "within_tol": False}
    d = {"loss_terms_vs_reference": [{"loss": ok, "policy": ok, "anchor": bad}],
         "logits_legal_vs_reference": [ok], "logits_padding_equal": [False], "values_vs_reference": [bad],
         "grads_vs_reference": [{"a.w": ok, "b.w": bad}],
         "repeats_bitwise": [{"loss_terms": True, "logits": False, "values": True, "grads": {"a.w": False}}],
         "additivity": {"a.w": bad}}
    assert E.judge_block(d) == ["evaluation 0: anchor", "evaluation 0: padded logits", "evaluation 0: values",
                                "evaluation 0: b.w", "repeat 1: logits", "repeat 1: a.w", "additivity: a.w"]


def test_COMPARE_adds_compact_evidence_only_outside_the_tolerance():
    s = {"cross_rel_tol": 1e-3, "cross_abs_tol": 1e-6}
    b = np.array([[1.0, -2.0], [0.5, 0.0]])
    assert E.compare(b.copy(), b, s) == {"bitwise": True, "max_abs": 0.0, "scale": 2.0, "within_tol": True}
    a = b.copy()
    a[1, 0] += 0.01
    c = E.compare(a, b, s)
    assert c["within_tol"] is False and c["n_outside"] == 1
    assert c["worst"] == {"index": [1, 0], "got": 0.51, "reference": 0.5}


def test_THE_VERDICT_is_PASS_only_with_all_three_phases_and_names_the_first_failure():
    p = {"status": "PASS"}
    assert E.verdict(dict.fromkeys(E.PHASES, p)) == {"verdict": "PASS", "failed_phase": None, "not_run": []}
    v = E.verdict({"policy_head": p, "full_model_incumbent": {"status": "FAIL"}})
    assert v == {"verdict": "FAIL", "failed_phase": "full_model_incumbent", "not_run": ["full_model_updated"]}
    assert E.verdict({"policy_head": {"status": "FAIL"}})["not_run"] == ["full_model_incumbent", "full_model_updated"]


# ─────────────────────────────── the public entry, in the REAL state ───────────────────────────────

def test_THE_PUBLIC_ENTRY_RUNS_ALL_THREE_PHASES_END_TO_END_IN_THE_REAL_STATE(opened, capsys):
    """The gate open, the pinned card, code and evidence, entry 26's fixtures, the real corpus (training-only reader)
    and the real incumbent, with the frozen spec. "gpu" is the CPU here, so every phase passes, the CPU repeats entry
    26 (prediction 1, and prediction 2's logits and policy term) and the prefix's state equals the CPU's own."""
    ckpt = _checkpoint()
    assert E.main(["run", "--checkpoint", str(ckpt)]) == E.EXIT_COMPLETED
    ev = opened / "evidence"
    rec = json.loads((ev / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "PASS", (rec.get("void"), rec.get("failed_phase"))
    assert rec["spec"] == json.loads(json.dumps(E.FROZEN)) and rec["checkpoint"]["sha1"] == M.CHECKPOINT_SHA1
    assert rec["entry26"]["policy_head"]["cpu_all_equal"] is True
    assert rec["entry26"]["full_model"]["cpu_prediction_holds"] is True
    held = {C._game_path(g) for g in rec["corpus"]["held_out_games_never_opened"]}      # the training-only reader
    opened_games = [f.split("/", 1)[1] for f in rec["corpus"]["files_opened"] if "/games/" in f]
    assert len(held) == 12 and len(opened_games) == 108 and not held & set(opened_games)
    assert rec["held_out_records_read"] == 0 and len(_lines(ev / "reconstruction.jsonl")) == 21
    assert rec["batches"]["0"] == json.loads((E.ROOT / E.ENTRY26 / "record.json").read_text())[
        "results"]["S0"]["batch"]
    assert sorted(p.name for p in (ev / "full").iterdir()) == ["S0_b0.json", "S0_b21.json", "U21_b0.json",
                                                              "U21_b21.json"]
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["verdict"] == "PASS" and out["phases"] == dict.fromkeys(E.PHASES, "PASS")
