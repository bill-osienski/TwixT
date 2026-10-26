"""C1, the composition-and-cost check's gated entry (`scripts/GPU/alphazero/mirror_halving_c1.py`).

The public entry is tested in the REAL state where it can be: the gate open (monkeypatched, never edited), the pinned
card, code and panel files (read-only), the real panel replay, the real prototypes, controls and cost totals. Only the
model is replaced: every evaluator is a GPU-free fake and entry 10's check is a stub, through the TEST seams.

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture redirects the evidence
directory and REPLACES `_real_seams` with fake seams, so nothing reached through the public entry can load a checkpoint
or touch a device. The fresh-process checks redirect the output and stub `conclude` too."""
import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from scripts.GPU.alphazero import mirror_halving_c1 as C1
from scripts.GPU.alphazero import mirror_halving_c1_authorization as GATE
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import search_distillation_d2_measure as M

REAL_SEAMS = C1._real_seams


class FakeEv:
    """30 zero channels; priors proportional to 1 + col (not mirror-symmetric); value 0.1. Its own network object."""

    def __init__(self):
        self.network = object()

    def build_input_tensor(self, state):
        return np.zeros((30, state.active_size, state.active_size), dtype=np.float32)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        w = (1.0 + move_cols.astype(np.float32)) * move_mask
        return (w / w.sum(axis=1, keepdims=True)).astype(np.float32), np.full(boards.shape[0], 0.1, np.float32)


def seams(**over):
    s = dict(versions=lambda: dict(C1.VERSIONS), checkpoint_sha1=lambda: M.CHECKPOINT_SHA1, load_evaluator=FakeEv,
             require_eval=lambda ev, when: {"modules_in_training_mode": 0, "when": when},
             entry10_check=lambda: {"verdict": "COMPLETED", "answer": "PASS"}, loaded_modules=lambda: {},
             read_bytes=lambda rel: (C1.ROOT / rel).read_bytes())
    s.update(over)
    return C1.Seams(**s)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(C1, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setattr(C1, "_real_seams", lambda ckpt: seams())
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(C1, "_mlx_loaded", lambda: False)          # this process may have MLX; subprocesses test it
    (tmp_path / "ckpt.safetensors").write_bytes(b"not a model")
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "MIRROR_HALVING_C1_AUTHORIZED", True)
    return isolated


def _run(tmp):
    return C1.main(["run", "--checkpoint", str(tmp / "ckpt.safetensors")])


def _record(tmp):
    return json.loads((tmp / "evidence" / "record.json").read_text())


# ─────────────────────────────── refusals ───────────────────────────────

def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.MIRROR_HALVING_C1_AUTHORIZED is False
    assert _run(isolated) == 4 and "is closed" in capsys.readouterr().err
    assert not (isolated / "evidence").exists()


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_MODEL_OR_PATHS(isolated):
    assert C1.EVIDENCE_DIR == str(isolated / "evidence")
    assert C1._real_seams is not REAL_SEAMS and type(C1._real_seams(None).load_evaluator()) is FakeEv


def test_ANYTHING_BUT_run_WITH_A_CHECKPOINT_IS_USAGE():
    for argv in ([], ["run"], ["run", "--checkpoint"], ["go", "--checkpoint", "/x"]):
        assert C1.main(argv) == 2


def test_THE_PINS_ARE_THE_FILES_ON_DISK():
    assert M._sha((C1.ROOT / C1.CARD).read_bytes()) == C1.CARD_SHA256
    for rel, sha in C1.PINNED.items():
        assert M._sha((C1.ROOT / rel).read_bytes()) == sha, rel


@pytest.mark.parametrize("pin", ["CARD_SHA256", "scripts/GPU/alphazero/mirror_averaging.py", C1.ENTRY10])
def test_AN_INPUT_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    if pin == "CARD_SHA256":
        monkeypatch.setattr(C1, pin, "0" * 64)
    else:
        monkeypatch.setitem(C1.PINNED, pin, "0" * 64)
    assert _run(opened) == 4 and "pinned" in capsys.readouterr().err and not (opened / "evidence").exists()


@pytest.mark.parametrize("value", [None, "1", "", " 0", "false"])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gate_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert _run(opened) == 4 and "MLX_ENABLE_TF32" in capsys.readouterr().err


def test_MLX_ALREADY_LOADED_REFUSES_even_with_the_gate_open(opened, monkeypatch, capsys):
    monkeypatch.setattr(C1, "_mlx_loaded", lambda: True)
    assert _run(opened) == 4 and "imported" in capsys.readouterr().err


def test_AN_EXISTING_EVIDENCE_DIRECTORY_OR_A_RELATIVE_CHECKPOINT_REFUSES(opened, capsys):
    assert C1.main(["run", "--checkpoint", "ckpt.safetensors"]) == 4
    (opened / "evidence").mkdir()
    assert _run(opened) == 4 and "one-shot" in capsys.readouterr().err
    assert list((opened / "evidence").iterdir()) == []


# ─────────────────────────────── the REAL setting check, in fresh processes ───────────────────────────────

PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import mirror_halving_c1 as C1
from scripts.GPU.alphazero import mirror_halving_c1_authorization as GATE
print("MLX_AFTER_IMPORT", C1._mlx_loaded(), flush=True)
t = pathlib.Path(tempfile.mkdtemp()); (t / "ckpt.safetensors").write_bytes(b"x")
C1.EVIDENCE_DIR = str(t / "evidence")
C1._real_seams = lambda ckpt: None
def stub(seams, out, head=None):       # never a real run, whatever a mutant does to the checks above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB"}
C1.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(C1.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    code = PRELUDE + body + "\nsys.exit(C1.main(['run', '--checkpoint', str(t / 'ckpt.safetensors')]))\n"
    return subprocess.run([sys.executable, "-c", code], cwd=C1.ROOT, env=env, capture_output=True, text=True,
                          timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("GATE.MIRROR_HALVING_C1_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("GATE.MIRROR_HALVING_C1_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_and_the_gate_open_every_real_check_passes():
    p = _fresh("GATE.MIRROR_HALVING_C1_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the public entry end to end ───────────────────────────────

def test_THE_PUBLIC_ENTRY_RUNS_EVERY_CHECK_ON_THE_REAL_PANEL_AND_PASSES(opened, capsys):
    assert _run(opened) == 0
    rec = _record(opened)
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "PASS" and rec["failed_checks"] == [], rec.get("void")
    assert rec["card_sha256"] == C1.CARD_SHA256 and rec["setting"] == rec["setting_at_end"] == {"MLX_ENABLE_TF32": "0"}
    assert rec["panel"]["positions"] == 104 and rec["check_rng"]["seed"] == 90009000
    assert len(rec["control_a"]) == 40 and all(r["equal"] for r in rec["control_a"])
    assert len(rec["control_b"]) == 64 and all(r["equal"] and r["last_choice"][0] for r in rec["control_b"])
    t = rec["cost_totals"]
    assert t[K.CANDIDATE]["searches"] == t[K.INCUMBENT]["searches"] == 104
    assert (t[K.CANDIDATE]["halving_searches"], t[K.INCUMBENT]["halving_searches"]) == (64, 0)
    g = rec["guards"]
    assert g["evaluations_ratio"] == t[K.CANDIDATE]["network_positions"] / t[K.INCUMBENT]["network_positions"]
    assert 0.95 < g["evaluations_ratio"] <= 1.05                      # 200 x 2 calls vs 400 x 1, roots included
    assert len(rec["cost_positions"]) == 104 and set(rec["derived"]["search_hours"]) == {"screen", "strength"}
    assert [json.loads(x)["event"] for x in (opened / "evidence" / "progress.jsonl").read_text().splitlines()][-1] \
        == "end"


def test_THE_COST_TOTALS_EXCLUDE_WARM_UP_CONTROLS_AND_THE_CORRECTNESS_CHECK(opened):
    """Every searched position's network positions add up to the totals exactly: nothing else is counted in them."""
    assert _run(opened) == 0
    rec = _record(opened)
    for k in (K.CANDIDATE, K.INCUMBENT):
        assert sum(r[k]["network_positions"] for r in rec["cost_positions"]) == \
            rec["cost_totals"][k]["network_positions"]
        assert [r["index"] for r in rec["cost_positions"]] == list(range(104))


def test_EVERY_FAILED_CHECK_IS_NAMED_AND_THE_VERDICT_IS_STOP(opened, monkeypatch):
    class Perturbed(K.PassThroughEvaluator):
        def infer(self, boards, move_rows, move_cols, move_mask, active_size):
            p, v = super().infer(boards, move_rows, move_cols, move_mask, active_size)
            return p, v + 1e-3

    monkeypatch.setattr(C1, "_real_seams", lambda ckpt: seams(entry10_check=lambda: {"verdict": "COMPLETED",
                                                                                     "answer": "FAIL"}))
    monkeypatch.setattr(K, "PassThroughEvaluator", Perturbed)
    monkeypatch.setitem(K.GUARDS, "evaluations", 0.5)
    assert _run(opened) == 0
    rec = _record(opened)
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "STOP"
    assert rec["failed_checks"] == ["entry10_correctness", "control_b_identical", "evaluations_within_guard"]


@pytest.mark.parametrize("over, step", [
    ({"versions": lambda: {"mlx": "0.30.4", "numpy": "2.4.1", "python": "3.14.9"}}, "versions"),
    ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint"),
    ({"read_bytes": lambda rel: b"{}"}, "panel"),
])
def test_A_WRONG_ENVIRONMENT_OR_INPUT_IS_A_VOID_BEFORE_ANY_SEARCH(opened, monkeypatch, over, step):
    calls = []
    monkeypatch.setattr(C1, "_real_seams", lambda ckpt: seams(load_evaluator=lambda: calls.append(1), **over))
    assert _run(opened) == 3
    rec = _record(opened)
    assert rec["status"] == "VOID" and rec["void"]["step"] == step and rec["verdict"] is None and calls == []


def test_TWO_SIDES_SHARING_A_NETWORK_IS_A_VOID(opened, monkeypatch):
    shared = FakeEv()
    monkeypatch.setattr(C1, "_real_seams", lambda ckpt: seams(load_evaluator=lambda: shared))
    assert _run(opened) == 3 and _record(opened)["void"]["step"] == "load"


SIGTERM_SCRIPT = """
import os, pathlib, signal, sys, tempfile, time
from tests.test_mirror_halving_c1 import seams
from scripts.GPU.alphazero import mirror_halving_c1 as C1
t = pathlib.Path(tempfile.mkdtemp())
def term():
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(5)
code, rec = C1.conclude(seams(entry10_check=term), t / "evidence")
print("RECORD", (t / "evidence" / "record.json").read_text().replace("\\n", " "), flush=True)
sys.exit(code)
"""


def test_SIGTERM_DURING_THE_RUN_IS_A_VOID_RECORD_in_a_fresh_isolated_process():
    """In a subprocess, so a defect that leaves SIGTERM unhandled kills that process, not pytest."""
    env = {**os.environ, "PYTHONPATH": str(C1.ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run([sys.executable, "-c", SIGTERM_SCRIPT], cwd=C1.ROOT, env=env, capture_output=True, text=True,
                       timeout=120)
    assert p.returncode == 3, p.stderr
    rec = json.loads(p.stdout.split("RECORD", 1)[1])
    assert rec["status"] == "VOID" and rec["void"]["step"] == "terminated" and rec["written_by"] == "script"
    assert signal.getsignal(signal.SIGTERM) is not None
