"""The full-float32 setting validation's gated entry (`scripts/GPU/alphazero/search_distillation_d2_tf32off.py`).

The public entry is tested in the REAL state: the gate open (monkeypatched, never edited), the pinned card and inputs,
the frozen probe SPEC and entry 27's real saved cases (read-only), with the "gpu" label remapped to the CPU and both
outputs redirected to tmp_path. No GPU computation runs here.

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture remaps the devices and
redirects both outputs for every in-process test. The checks of the REAL setting and of MLX already being loaded need
a fresh process; those subprocesses redirect the outputs and replace `conclude` with a stub, so even a mutant that
removes every check before it cannot reach a real run."""
import io
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from scripts.GPU.alphazero import mlx_matmul_rows_probe as P
from scripts.GPU.alphazero import search_distillation_d2_tf32off as E
from scripts.GPU.alphazero import search_distillation_d2_tf32off_authorization as GATE


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    import mlx.core as mx
    monkeypatch.setattr(P, "DEVICES", {"gpu": mx.cpu, "cpu": mx.cpu})
    monkeypatch.setattr(E, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(E, "OUTPUT_PARENT", tmp_path / "cache")
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(E, "_mlx_loaded", lambda: False)          # this process has MLX; the subprocesses test it
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED", True)
    return isolated


def _record(tmp):
    return json.loads((tmp / "evidence" / "record.json").read_text())


# ─────────────────────────────── refusals ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED is False
    assert E.main(["run"]) == 4
    assert "is closed" in capsys.readouterr().err
    assert not (isolated / "evidence").exists() and list((isolated / "cache").iterdir()) == []


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_DEVICES_OR_PATHS(isolated):
    import mlx.core as mx
    assert P.DEVICES == {"gpu": mx.cpu, "cpu": mx.cpu}
    assert E.OUTPUT_PARENT == isolated / "cache" and E.EVIDENCE_DIR == str(isolated / "evidence")


def test_ANYTHING_BUT_run_IS_USAGE():
    assert E.main([]) == 2 and E.main(["run", "--x"]) == 2


def test_THE_PINS_ARE_THE_FILES_ON_DISK():
    card = (E.ROOT / E.CARD).read_text()
    assert E._sha((E.ROOT / E.CARD).read_bytes()) == E.CARD_SHA256
    for rel, sha in ((E.GENERATOR, E.GENERATOR_SHA256), (E.LINREPRO, E.LINREPRO_SHA256),
                     (E.FIXTURE_ARTIFACTS, E.FIXTURE_ARTIFACTS_SHA256), (E.ENTRY27_CASES, E.ENTRY27_CASES_SHA256),
                     (E.ENTRY30_CASES, E.ENTRY30_CASES_SHA256)):
        assert E._sha((E.ROOT / rel).read_bytes()) == sha and sha in card, rel
    assert E.GENERATOR_SHA256 == "1217bc8e48589b077bbe0962564ffb5c0298326f96de6658ae39a83cf739bb57"   # unchanged probe


@pytest.mark.parametrize("pin", ["CARD_SHA256", "GENERATOR_SHA256", "LINREPRO_SHA256", "FIXTURE_ARTIFACTS_SHA256",
                                 "ENTRY27_CASES_SHA256", "ENTRY30_CASES_SHA256"])
def test_AN_INPUT_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    monkeypatch.setattr(E, pin, "0" * 64)
    assert E.main(["run"]) == 4 and "pinned" in capsys.readouterr().err
    assert not (opened / "evidence").exists()


@pytest.mark.parametrize("value", [None, "1", "", " 0", "false"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert E.main(["run"]) == 4 and "MLX_ENABLE_TF32" in capsys.readouterr().err
    assert not (opened / "evidence").exists()


def test_MLX_ALREADY_LOADED_REFUSES_even_with_the_gate_open(opened, monkeypatch, capsys):
    monkeypatch.setattr(E, "_mlx_loaded", lambda: True)
    assert E.main(["run"]) == 4 and "imported" in capsys.readouterr().err
    assert not (opened / "evidence").exists()


def test_AN_EXISTING_EVIDENCE_DIRECTORY_REFUSES_untouched(opened, capsys):
    (opened / "evidence").mkdir()
    assert E.main(["run"]) == 4 and "one-shot" in capsys.readouterr().err
    assert list((opened / "evidence").iterdir()) == []


def test_A_MISSING_OUTPUT_PARENT_OR_FIXTURE_DIRECTORY_REFUSES(opened, monkeypatch):
    monkeypatch.setattr(E, "OUTPUT_PARENT", opened / "nowhere")
    assert E.main(["run"]) == 4 and not (opened / "evidence").exists()
    monkeypatch.setattr(E, "OUTPUT_PARENT", opened / "cache")
    monkeypatch.setattr(E, "FIXTURE_DIR", opened / "nowhere")
    assert E.main(["run"]) == 4 and not (opened / "evidence").exists()


# ─────────────────────────────── the REAL setting check, in fresh processes ───────────────────────────────

PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import search_distillation_d2_tf32off as E
from scripts.GPU.alphazero import search_distillation_d2_tf32off_authorization as GATE
print("MLX_AFTER_IMPORT", E._mlx_loaded(), flush=True)
t = pathlib.Path(tempfile.mkdtemp()); (t / "cache").mkdir()
E.EVIDENCE_DIR = str(t / "evidence"); E.OUTPUT_PARENT = t / "cache"
def stub(out, example):             # never a real run, whatever a mutant does to the checks above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB"}
E.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(E.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    return subprocess.run([sys.executable, "-c", PRELUDE + body + "\nsys.exit(E.main(['run']))\n"], cwd=E.ROOT,
                          env=env, capture_output=True, text=True, timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the public entry end to end ───────────────────────────────

def test_THE_PUBLIC_ENTRY_RUNS_BOTH_SETS_END_TO_END(opened, capsys):
    """The real state: gate open, pinned card and inputs, the frozen SPEC and entry 27's real saved cases; 'gpu' is
    the CPU here, so every count is zero, the verdict is CANDIDATE and the CPU repeats entries 27 and 30 exactly."""
    assert E.main(["run"]) == 0
    ev = opened / "evidence"
    rec = _record(opened)
    assert rec["status"] == "COMPLETED" and rec["written_by"] == "script"
    assert rec["setting"] == rec["setting_at_end"] == {"MLX_ENABLE_TF32": "0"}
    probe = json.loads((ev / "probe" / "record.json").read_text())
    assert probe["status"] == "COMPLETED" and probe["spec"] == P.SPEC and len(probe["case_list"]) == 27
    assert rec["probe"]["status"] == "COMPLETED"
    lines = [json.loads(x) for x in (ev / "fixture_cases.jsonl").read_text().splitlines()]
    assert [c["id"] for c in lines] == E.FROZEN["fixture_cases"] == rec["fixture_cases_done"]
    for c in lines:
        assert list(c["implementations"]) == E.FROZEN["implementations"]
        for per in c["implementations"].values():
            assert [len(per[d]["outputs"]["Y"]) for d in ("gpu", "cpu")] == [3, 1]
    for s, n_gpu in (("synthetic", 27 * 4 * 3), ("fixtures", 4 * 3 * 4 * 3)):
        assert rec["counts"][s]["gpu"]["evaluations"] == n_gpu
        assert all(rec["counts"][s][d][k] == 0 for d in ("gpu", "cpu") for k in ("T", "G", "R")), rec["counts"][s]
    assert rec["verdict"] == "CANDIDATE" and rec["sets_pass"] == {"synthetic": True, "fixtures": True}
    for name, total in (("synthetic_vs_entry30", 27 * 4), ("fixtures_vs_entry27", 4 * 3 * 4)):
        cpu = rec["comparisons"][name]["cpu"]
        assert cpu == {"equal": total, "different": 0, "different_keys": []}, (name, cpu)
    assert list((opened / "cache").iterdir()) == []                     # nothing gross: no example directory
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "COMPLETED" and out["verdict"] == "CANDIDATE"


def test_A_FIXTURE_THAT_IS_NOT_THE_RECORDED_FILE_IS_VOID_after_set_A(opened, monkeypatch):
    real = E._read_fixture
    monkeypatch.setattr(E, "_read_fixture", lambda name: b"x" if name == "out_S0_original" else real(name))
    assert E.main(["run"]) == 3
    rec = _record(opened)
    assert rec["status"] == "VOID" and rec["void"]["step"] == "fixtures" and "out_S0_original" in rec["void"]["detail"]
    assert rec["probe"]["status"] == "COMPLETED" and rec["fixture_cases_done"] == ["fc_S0_original", "fc_S21_original"]
    assert "verdict" not in rec


def test_SAVED_EXPECTED_ARRAYS_MUST_EQUAL_THE_RECOMPUTATION(isolated, monkeypatch):
    rng = np.random.default_rng(0)
    o = {"X": rng.standard_normal((40, 2)).astype(np.float32), "W": rng.standard_normal((3, 2)).astype(np.float32),
         "b": rng.standard_normal(3).astype(np.float32), "C": rng.standard_normal((40, 3)).astype(np.float32),
         "legal": np.ones(40, bool)}
    ref = P.expected(o)
    monkeypatch.setattr(E, "FIXTURE_DIR", isolated)
    for name, bump in (("fc_S0_original", 0.0), ("fc_S21_original", 1e-12)):
        buf = io.BytesIO()
        np.savez(buf, **{f"operand.{k}": v for k, v in o.items()},
                 **{f"expected.{k}": v + (bump if k == "dW" else 0.0) for k, v in ref.items()})
        (isolated / f"{name}.npz").write_bytes(buf.getvalue())
    entries = {n: {"path": str(isolated / f"{n}.npz"), "sha256": E._sha((isolated / f"{n}.npz").read_bytes())}
               for n in ("fc_S0_original", "fc_S21_original")}
    got, again, source = E.fixture_operands("fc_S0_original", entries)
    assert all(np.array_equal(again[k], ref[k]) for k in ref) and source == entries["fc_S0_original"]
    with pytest.raises(E.Void, match="saved expected dW differs"):
        E.fixture_operands("fc_S21_original", entries)
    with pytest.raises(E.Void, match="not recorded"):
        E.fixture_operands("fc_S0_original", {"fc_S0_original": {**entries["fc_S0_original"], "path": "/elsewhere"}})


def test_A_SET_A_THAT_DOES_NOT_COMPLETE_IS_VOID_AND_SET_B_NEVER_RUNS(opened, monkeypatch):
    def probe(out, example, spec):
        out.mkdir()
        return 3, {"status": "VOID", "void": {"step": "non_finite", "detail": "x"}}
    monkeypatch.setattr(P, "execute", probe)
    monkeypatch.setattr(E, "fixture_case", lambda *a: pytest.fail("set B ran after an incomplete set A"))
    assert E.main(["run"]) == 3
    rec = _record(opened)
    assert rec["status"] == "VOID" and rec["void"]["step"] == "probe" and rec["probe"]["status"] == "VOID"
    assert not (opened / "evidence" / "fixture_cases.jsonl").exists()


SIGTERM_BODY = """
import os, signal, time
import mlx.core as mx
from scripts.GPU.alphazero import mlx_matmul_rows_probe as P
P.DEVICES.update({"gpu": mx.cpu, "cpu": mx.cpu})
E._mlx_loaded = lambda: False
E.conclude = REAL_CONCLUDE
GATE.SEARCH_DISTILLATION_D2_TF32OFF_AUTHORIZED = True
def term(*a):
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(5)
E.fixture_case = term
print("EVIDENCE", E.EVIDENCE_DIR, flush=True)
"""


def test_SIGTERM_DURING_SET_B_IS_A_VOID_RECORD_in_a_fresh_isolated_process():
    """In a subprocess, so a defect that leaves SIGTERM unhandled kills that process, not pytest."""
    body = "REAL_CONCLUDE = E.conclude\n" + SIGTERM_BODY
    script = PRELUDE.replace("E.conclude = stub\n", "") + body
    env = {**os.environ, "PYTHONPATH": str(E.ROOT), "PYTHONDONTWRITEBYTECODE": "1", "MLX_ENABLE_TF32": "0"}
    p = subprocess.run([sys.executable, "-c", script + "\nsys.exit(E.main(['run']))\n"], cwd=E.ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 3, (p.returncode, p.stderr[-2000:])
    ev = pathlib.Path(next(x for x in p.stdout.splitlines() if x.startswith("EVIDENCE ")).split(" ", 1)[1])
    rec = json.loads((ev / "record.json").read_text())
    assert rec["status"] == "VOID" and rec["void"]["step"] == "terminated" and rec["probe"]["status"] == "COMPLETED"
    assert not rec["example_dir"].startswith(str(pathlib.Path.home() / "twixt-cache"))      # redirected


# ─────────────────────────────── counts, verdict, comparisons ───────────────────────────────

def _j(outside, gross, rows_out=None, rows_gross=None, max_abs=0.0):
    j = {"outside_tol": outside, "gross": gross, "max_abs": max_abs}
    if rows_out is not None:
        j.update(rows_outside_tol=rows_out, rows_gross=rows_gross)
    return j


def test_COUNTS_separate_tolerance_gross_and_repeatability_and_tolerance_only_rows():
    d = {"outputs": {"Y": [_j(3, 0, 2, 0), _j(5, 1, 4, 1), _j(0, 0, 0, 0)], "dW": [_j(0, 0)] * 3},
         "repeats_bitwise": {"Y": [True, False], "dW": [True, True]}}
    c = E.counts([("fc_33600", "fc", "probe", "gpu", d)])["gpu"]
    assert (c["evaluations"], c["T"], c["G"], c["R"]) == (6, 2, 1, 1)
    assert c["T_keys"] == ["fc_33600/probe/Y/0", "fc_33600/probe/Y/1"] and c["G_keys"] == ["fc_33600/probe/Y/1"]
    assert c["R_keys"] == ["fc_33600/probe/Y"]
    assert (c["tolerance_only_rows"], c["tolerance_only_row_evaluations"]) == (5, 2)


def _per_set(**gpu):
    zero = {"T": 0, "G": 0, "R": 0}
    return {s: {"gpu": {**zero, **gpu.get(s, {})}, "cpu": {"T": 9, "G": 9, "R": 9}} for s in ("synthetic", "fixtures")}


def test_THE_VERDICT_needs_every_GPU_count_zero_in_BOTH_sets_and_ignores_the_CPU():
    assert E.verdict(_per_set())["verdict"] == "CANDIDATE"                  # CPU counts of 9 never enter
    v = E.verdict(_per_set(synthetic={"T": 4}))
    assert v["verdict"] == "NOT_A_WORKAROUND" and v["failed_counts"] == {"synthetic": ["T"], "fixtures": []}
    assert v["gross_removed_but_tolerance_failing"] == {"synthetic": True, "fixtures": False}
    v = E.verdict(_per_set(fixtures={"R": 1}))
    assert v["verdict"] == "NOT_A_WORKAROUND" and v["sets_pass"] == {"synthetic": True, "fixtures": False}
    v = E.verdict(_per_set(synthetic={"T": 2, "G": 2}))
    assert v["failed_counts"]["synthetic"] == ["T", "G"] and not v["gross_removed_but_tolerance_failing"]["synthetic"]


def test_COMPARISONS_group_the_ungated_outputs_and_count_exact_equality():
    gpu = {"outputs": {"Y": [_j(0, 0, max_abs=1.0)], "dX": [_j(0, 0, max_abs=2.0)], "dW": [_j(0, 0, max_abs=3.0)],
                       "db": [_j(0, 0, max_abs=4.0)]}, "repeats_bitwise": {}}
    cpu = {"outputs": {"Y": [_j(0, 0, max_abs=5.0)]}, "repeats_bitwise": {}}
    prior = {("fc", "gpu", "Y"): 1.0, ("fc", "gpu", "dX"): 2.5, ("fc", "gpu", "dW"): 3.0, ("out", "gpu", "Y"): 1.0,
             ("out", "gpu", "dW"): 3.0, ("out", "gpu", "db"): 4.0, ("fc", "cpu", "Y"): 5.0}
    rows = [("fc", "fc", "probe", "gpu", gpu), ("fc", "fc", "probe", "cpu", cpu), ("out", "out", "probe", "gpu", gpu)]
    c = E.compare(rows, lambda case, impl, dev, k, i: prior.get((case, dev, k)))
    assert c["cpu"] == {"equal": 1, "different": 0, "different_keys": []}
    assert c["gpu_ungated"] == {"equal": 4, "different": 1, "different_keys": ["fc/probe/gpu/db/0"]}  # fc db: missing
    assert c["gpu_gated"] == {"equal": 1, "different": 2, "different_keys": ["fc/probe/gpu/dX/0",
                                                                             "out/probe/gpu/dX/0"]}


def test_A_LINREPRO_MODULE_THAT_IS_NOT_THE_PINNED_FILE_IS_VOID(opened, monkeypatch):
    """The pin passes for whatever file LINREPRO names; the imported module must also BE that file."""
    monkeypatch.setattr(E, "LINREPRO", E.GENERATOR)
    monkeypatch.setattr(E, "LINREPRO_SHA256", E.GENERATOR_SHA256)
    assert E.main(["run"]) == 3
    rec = _record(opened)
    assert rec["status"] == "VOID" and rec["void"]["step"] == "linrepro" and rec["fixture_cases_done"] == []
