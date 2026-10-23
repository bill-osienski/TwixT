"""The synthetic row-count probe's gated entry (`scripts/GPU/alphazero/search_distillation_d2_mlxrows.py`).

The public entry is tested in the REAL state: the gate open (monkeypatched, never edited), the pinned card and
generator, the frozen SPEC end to end, with the "gpu" label remapped to the CPU and both outputs redirected to
tmp_path. No GPU computation runs here.

🔴 EVERY TEST IS ISOLATED, NOT ONLY THE ONES THAT OPEN THE GATE. The first build isolated only the gate-open tests.
Under the injected defect "gate unchecked", the closed-gate test reached the real entry with the real devices and
paths: the full frozen probe ran on the GPU inside the control worktree and left a compact example in the real
~/twixt-cache (ledger, post-closure entry 28). So the autouse fixture below remaps the devices to the CPU and
redirects both outputs for every test, whatever the gate says."""
import json
import re

import pytest

from scripts.GPU.alphazero import mlx_matmul_rows_probe as P
from scripts.GPU.alphazero import search_distillation_d2_mlxrows as E
from scripts.GPU.alphazero import search_distillation_d2_mlxrows_authorization as GATE


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    import mlx.core as mx
    monkeypatch.setattr(P, "DEVICES", {"gpu": mx.cpu, "cpu": mx.cpu})
    monkeypatch.setattr(E, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(E, "OUTPUT_PARENT", tmp_path / "cache")
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(GATE, "SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED", True)
    return isolated


def test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert GATE.SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED is False
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
    assert E._sha(E.ROOT / E.CARD) == E.CARD_SHA256
    assert E._sha(E.ROOT / E.GENERATOR) == E.GENERATOR_SHA256
    assert E.GENERATOR_SHA256 in (E.ROOT / E.CARD).read_text()


@pytest.mark.parametrize("pin", ["CARD_SHA256", "GENERATOR_SHA256"])
def test_A_CARD_OR_GENERATOR_THAT_IS_NOT_THE_PINNED_BYTES_REFUSES(opened, monkeypatch, pin, capsys):
    monkeypatch.setattr(E, pin, "0" * 64)
    assert E.main(["run"]) == 4 and "pinned" in capsys.readouterr().err
    assert not (opened / "evidence").exists()


def test_AN_EXISTING_EVIDENCE_DIRECTORY_REFUSES_untouched(opened, capsys):
    (opened / "evidence").mkdir()
    assert E.main(["run"]) == 4 and "one-shot" in capsys.readouterr().err
    assert list((opened / "evidence").iterdir()) == []


def test_A_MISSING_OUTPUT_PARENT_REFUSES(opened, monkeypatch):
    monkeypatch.setattr(E, "OUTPUT_PARENT", opened / "nowhere")
    assert E.main(["run"]) == 4 and not (opened / "evidence").exists()


def test_THE_ENTRY_HANDS_THE_GENERATOR_EXACTLY_ITS_TWO_DIRECTORIES(opened, monkeypatch):
    seen = []
    monkeypatch.setattr(P, "main", lambda argv: seen.append(argv) or 0)
    assert E.main(["run"]) == 0
    (argv,) = seen
    assert argv[:3] == ["--out", str(opened / "evidence"), "--example-dir"]
    assert re.fullmatch(re.escape(str(opened / "cache")) + r"/mlxrows-\d{8}T\d{6}Z", argv[3]) and len(argv) == 4


def test_THE_PUBLIC_ENTRY_RUNS_THE_FROZEN_CASE_LIST_END_TO_END(opened, capsys):
    """The real state: gate open, pinned card and generator, the frozen SPEC; 'gpu' is the CPU here, so nothing is
    gross, no example is saved and the example directory is never created."""
    assert E.main(["run"]) == 0
    ev = opened / "evidence"
    rec = json.loads((ev / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["spec"] == P.SPEC
    assert len(rec["case_list"]) == 27 == len((ev / "cases.jsonl").read_text().splitlines())
    assert rec["case_list"][0] == "fc_32704" and rec["case_list"][-1] == "wide_33728"
    assert rec["cases_with_gpu_gross"] == rec["cases_with_cpu_gross"] == [] and rec["example"] is None
    assert json.loads((ev / "environment.json").read_text())["generator_sha256"] == E.GENERATOR_SHA256
    assert list((opened / "cache").iterdir()) == []
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "COMPLETED"
