"""H-MX export attempt 2: model-free tests (docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md).

No model loads and no gate opens in this repository. The entry runs on a fake runner in a fixture tree; the attempt-1
binding and the Node dependency are also checked against the REAL committed state; the wrapper's Node preflight runs
for real (bash, git and node) in a throwaway repository, with a fake interpreter that records any call.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

import pytest

from scripts.GPU.alphazero import htb_mix as M
from scripts.GPU.alphazero import htb_mix_export as XM
from scripts.GPU.alphazero import htb_pilot as P
from tests import htb_mix_export_2_wrapper_spec as W2
from tests import test_htb_mix as TM

ROOT = pathlib.Path(__file__).resolve().parents[1]
sha = lambda b: hashlib.sha256(b).hexdigest()                                    # noqa: E731
A1 = XM.ATTEMPT_1
R_EV_RECORD = "docs/superpowers/evidence/htb-mix-recovery/record.json"


# ─────────────────────────────── the pins, against the real committed state ───────────────────────────────

def test_THE_AMENDMENT_IS_PINNED_IN_PYTHON_AND_IN_THE_ANALYSER():
    raw = (ROOT / XM.AMENDMENT2).read_bytes()
    assert sha(raw) == XM.AMENDMENT2_SHA256
    assert b"**Status:** DRAFT for review (2026-11-20)" in raw
    ana = (ROOT / "tests/htb_integration/mix_match_analyse.mjs").read_text()
    assert f"export const EXPORT_AMENDMENT_SHA256 = '{XM.AMENDMENT2_SHA256}';" in ana
    assert "export const EXPORT_ATTEMPT = 2;" in ana


def test_ATTEMPT_1_IS_PRESERVED_AS_COMMITTED_AND_IS_VOID():
    for rel, want in ((A1["record"], A1["record_sha256"]), (A1["outer"], A1["outer_sha256"])):
        assert sha((ROOT / rel).read_bytes()) == want
        assert subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
        assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    rec = json.loads((ROOT / A1["record"]).read_text())
    assert rec["status"] == "VOID" and "verdict" not in rec and "attempt" not in rec
    assert json.loads((ROOT / A1["outer"]).read_text())["outcome"] == "CRASHED"
    assert (A1["record"], A1["outer"]) == (f"{M.EV['export']}/record.json", f"{M.OUTER['export']}/outer.json")


def test_ATTEMPT_2_WRITES_ONLY_UNDER_ITS_OWN_FRESH_DIRECTORIES():
    assert (XM.EV2, XM.OUTER2) == ("docs/superpowers/evidence/htb-mix-export-2", "docs/superpowers/evidence/htb-mix-export-2-run")
    cfg = XM.Attempt2Config()
    assert cfg.ev("export") == ROOT / XM.EV2 and cfg.ev("fit") == ROOT / M.EV["fit"]
    assert M.Config().ev("export") == ROOT / M.EV["export"]                     # attempt 1's own entry is unchanged
    assert not (ROOT / XM.EV2).exists()
    assert sorted(p.name for p in (ROOT / XM.OUTER2).iterdir()) == ["01_wrapper.sh"]
    assert (XM.ORT_NODE_VERSION, M.KILL_S["export"]) == ("1.23.2", 3_600)


def test_THE_REAL_BINDINGS_HOLD_IN_THIS_CHECKOUT():
    """Read-only: the amendment, attempt 1's committed VOID evidence and the main checkout's onnxruntime-node."""
    cfg = XM.Attempt2Config(gate_open=lambda s: True, tf32_required=False)
    extra = XM.attempt_2_bindings(cfg)
    assert extra["attempt"] == 2 and extra["export_amendment_sha256"] == XM.AMENDMENT2_SHA256
    assert extra["attempt_1"] == A1
    assert extra["node_dependency"] == {"onnxruntime_node": "1.23.2", "package": str(ROOT / XM.ORT_PACKAGE)}


# ─────────────────────────────── the entry, on fakes ───────────────────────────────

def _node_modules(root, version="1.23.2"):
    pkg = root / XM.ORT_PACKAGE
    pkg.parent.mkdir(parents=True, exist_ok=True)
    pkg.write_text(json.dumps({"name": "onnxruntime-node", "version": version}))


def _attempt2(tmp_path, **over):
    """A fixture tree: the card, the amendment, attempt 1's REAL committed VOID evidence, a clean verified recovery,
    the current pair and a node_modules holding onnxruntime-node 1.23.2."""
    root = TM._tree(tmp_path)
    TM._current(root)
    for rel in (XM.AMENDMENT2, A1["record"], A1["outer"]):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    _node_modules(root)
    base = TM._cfg(root)
    cfg = XM.Attempt2Config(**{f: getattr(base, f) for f in ("root", "data_root", "cache", "committed",
                                                             "evidence_problems", "head", "gate_open", "exclusions",
                                                             "tf32_required")}, **over)
    TM._recovery_evidence(cfg, tmp_path)
    return root, cfg


def _bytes(root, rels):
    return {rel: (root / rel).read_bytes() for rel in rels}


def test_ATTEMPT_2_RUNS_THE_UNCHANGED_EXPORT_INTO_FRESH_EVIDENCE_AND_LEAVES_ATTEMPT_1_AS_WRITTEN(tmp_path):
    root, cfg = _attempt2(tmp_path)
    before = _bytes(root, (A1["record"], A1["outer"]))
    run, calls = TM._fake_runner()
    assert XM.run_export_2(cfg, runner=run, **TM._xargs(tmp_path)) == 0
    rec = json.loads((root / XM.EV2 / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["verdict"] == "PARITY_PASS" and rec["kind"] == "htb-mix-export-record/1"
    assert rec["attempt"] == 2 and rec["export_amendment_sha256"] == XM.AMENDMENT2_SHA256 and rec["attempt_1"] == A1
    assert rec["node_dependency"]["onnxruntime_node"] == "1.23.2" and rec["kill_s"] == 3_600
    assert calls == ["export", "canary_python", "canary_node", "s_python", "s_node", "s_compare", "v3_python", "v3_node",
                     "v3_compare"]                                         # the card's steps, unchanged
    assert _bytes(root, (A1["record"], A1["outer"])) == before
    assert sorted(p.name for p in (root / M.EV["export"]).iterdir()) == ["record.json"]
    with pytest.raises(P.Refused, match="already exists"):                  # one-shot
        XM.run_export_2(cfg, runner=run, **TM._xargs(tmp_path))


def test_ATTEMPT_2_IS_ONE_SHOT_AND_ATTEMPT_1S_ENTRY_STILL_REFUSES(tmp_path):
    root, cfg = _attempt2(tmp_path)
    run, calls = TM._fake_runner()
    (root / XM.EV2).mkdir(parents=True)
    with pytest.raises(P.Refused, match="already exists"):
        XM.run_export_2(cfg, runner=run, **TM._xargs(tmp_path))
    with pytest.raises(P.Refused, match="already exists"):                  # attempt 1's evidence is there
        XM.run_export(TM._cfg(root), runner=run, **TM._xargs(tmp_path))
    assert calls == []


@pytest.mark.parametrize("fault", ["gate", "amendment", "a1_record_changed", "a1_record_missing", "a1_outer_changed",
                                   "a1_not_void", "no_node_modules", "symlinked_node_modules", "no_package",
                                   "old_version", "unparseable_package", "recovery"])
def test_EVERY_BINDING_REFUSES_BEFORE_ANYTHING_IS_WRITTEN_OR_RUN(tmp_path, fault):
    root, cfg = _attempt2(tmp_path)
    if fault == "gate":
        cfg.gate_open = lambda s: False
    elif fault == "amendment":
        (root / XM.AMENDMENT2).write_bytes((root / XM.AMENDMENT2).read_bytes() + b"\n")
    elif fault == "a1_record_changed":
        (root / A1["record"]).write_bytes((root / A1["record"]).read_bytes().replace(b"VOID", b"COMPLETED"))
    elif fault == "a1_record_missing":
        (root / A1["record"]).unlink()
    elif fault == "a1_outer_changed":
        (root / A1["outer"]).write_bytes(b"{}")
    elif fault == "a1_not_void":                    # a record pinned by hash whose status is not VOID
        raw = (root / A1["record"]).read_bytes().replace(b'"VOID"', b'"COMPLETED"')
        (root / A1["record"]).write_bytes(raw)
        cfg.attempt_1 = {**A1, "record_sha256": sha(raw)}
    elif fault == "no_node_modules":
        shutil.rmtree(root / "node_modules")
    elif fault == "symlinked_node_modules":
        shutil.move(root / "node_modules", tmp_path / "elsewhere")
        (root / "node_modules").symlink_to(tmp_path / "elsewhere")
    elif fault == "no_package":
        (root / XM.ORT_PACKAGE).unlink()
    elif fault == "old_version":
        _node_modules(root, "1.23.1")
    elif fault == "unparseable_package":
        (root / XM.ORT_PACKAGE).write_text("{")
    else:
        (root / R_EV_RECORD).write_text(json.dumps({"status": "FAILED"}))
    want = {"gate": "HTB_MIX_EXPORT_AUTHORIZED is closed", "amendment": "not the pinned file",
            "a1_record_changed": "not the pinned file", "a1_record_missing": "absent, uncommitted or changed",
            "a1_outer_changed": "not the pinned file", "a1_not_void": "not attempt 1's VOID record",
            "no_node_modules": "not a real directory", "symlinked_node_modules": "not a real directory",
            "no_package": "onnxruntime-node is not installed", "old_version": "onnxruntime-node 1.23.1",
            "unparseable_package": "unreadable", "recovery": "not a COMPLETED recovery"}
    run, calls = TM._fake_runner()
    xargs = TM._xargs(tmp_path)
    with pytest.raises(P.Refused, match=want[fault]):
        XM.run_export_2(cfg, runner=run, **xargs)
    assert calls == [] and not (root / XM.EV2).exists()
    assert not xargs["export_root"].exists() and not xargs["scratch"].exists()



def test_THE_RECOVERY_ADMISSION_STILL_READS_ITS_OWN_AMENDMENT_NOT_THIS_ONE(tmp_path):
    """Attempt2Config names its pin `export_amendment_sha256`: R.admission reads `amendment_sha256` as the recovery's."""
    root, cfg = _attempt2(tmp_path)
    assert not hasattr(cfg, "amendment_sha256")
    assert XM.recovery_admission(cfg)["challenger"]["sha256"]


def test_THE_CLI_OFFERS_EXPORT_2_AND_REFUSES_WHILE_THE_GATE_IS_CLOSED(capsys):
    snap = {d: sorted(p.name for p in (ROOT / d).iterdir()) if (ROOT / d).exists() else None
            for d in (M.EV["export"], M.OUTER["export"], XM.EV2, XM.OUTER2)}
    assert XM.main(["run", "export-2"]) == P.EXIT["REFUSED"]
    assert "HTB_MIX_EXPORT_AUTHORIZED is closed" in capsys.readouterr().err
    assert snap == {d: sorted(p.name for p in (ROOT / d).iterdir()) if (ROOT / d).exists() else None for d in snap}
    with pytest.raises(SystemExit):
        XM.main(["run", "export-3"])


# ─────────────────────────────── the wrapper ───────────────────────────────

def test_THE_WRAPPER_IS_ITS_DERIVATION_WITH_THE_PREFLIGHT_BEFORE_THE_IDLE_GATE():
    text = W2.wrapper_path().read_text()
    assert text == W2.derive()
    body = text.split("set -u\n", 1)[1]
    assert "OUT_REL=docs/superpowers/evidence/htb-mix-export-2-run\n" in body
    assert "RUN_REL=docs/superpowers/evidence/htb-mix-export-2\n" in body
    assert "TERM_S=3590\n" in body and "GRACE_S=10\n" in body and "GATE=HTB_MIX_EXPORT_AUTHORIZED\n" in body
    assert '-m "$MODULE" run export-2 \\\n' in body and "run export \\" not in body
    pre = body.index("# THE NODE DEPENDENCY")
    assert body.index('[ -e "$RUN_REL" ] && refuse') < pre < body.index("# THE IDLE GATE") < body.index("started=true")
    assert 'the amendment pins 1.23.2"' in body
    parent = (ROOT / W2.PARENT).read_text().split("set -u\n", 1)[1]
    assert body.replace(W2.PREFLIGHT, "").replace("export-2", "export") == parent
    assert subprocess.run(["bash", "-n", str(W2.wrapper_path())]).returncode == 0


GATE_REL = "scripts/GPU/alphazero/htb_mix_export_authorization.py"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _wrapper_repo(tmp_path, case):
    """A throwaway repository holding the real derived wrapper, a closed-then-opened gate (one-line commit), a fake
    interpreter that logs and fails, and node_modules as `case` says."""
    w = tmp_path / "w"
    wrapper = w / XM.OUTER2 / "01_wrapper.sh"
    wrapper.parent.mkdir(parents=True)
    shutil.copyfile(W2.wrapper_path(), wrapper)
    wrapper.chmod(0o755)
    (w / GATE_REL).parent.mkdir(parents=True)
    (w / GATE_REL).write_text('"""gate"""\nHTB_MIX_EXPORT_AUTHORIZED = False\n')
    (w / "server").mkdir()
    (w / "server/inference.js").write_text("// fixture\n")
    (w / ".gitignore").write_text("node_modules\n.venv\n")
    log = tmp_path / "py_calls.txt"
    py = w / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit 1\n')
    py.chmod(0o755)
    _git(w, "init", "-q")
    _git(w, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(w, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "closed")
    (w / GATE_REL).write_text('"""gate"""\nHTB_MIX_EXPORT_AUTHORIZED = True\n')
    _git(w, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "open")
    if case == "ok":
        _node_modules(w)
    elif case == "old":
        _node_modules(w, "1.23.1")
    elif case == "symlink":
        _node_modules(tmp_path / "other")
        (w / "node_modules").symlink_to(tmp_path / "other/node_modules")
    elif case == "empty":
        (w / "node_modules").mkdir()
    home = tmp_path / "home"
    (home / "twixt-cache/htb-pilot").mkdir(parents=True)
    (home / "twixt-cache/htb-pilot/student.safetensors").write_bytes(b"x")
    r = subprocess.run(["bash", str(wrapper)], env={**os.environ, "HOME": str(home)}, capture_output=True, text=True,
                       timeout=120)
    return w, r, log


@pytest.mark.parametrize("case, refusal", [
    ("missing", "node_modules is not a real directory in this tree"),
    ("symlink", "node_modules is not a real directory in this tree"),
    ("empty", r"Node cannot resolve onnxruntime-node from this tree|not this tree's node_modules"),
    ("old", "onnxruntime-node 1.23.1, the amendment pins 1.23.2"),
    ("ok", r"the machine did not become idle"),
])
def test_THE_REAL_WRAPPER_PREFLIGHT_REFUSES_CLEANLY_ON_A_MISSING_OR_WRONG_DEPENDENCY(tmp_path, case, refusal):
    """Run for real: a miss refuses at the preflight (the interpreter is never called, nothing is written, the gate
    is restored). With 1.23.2 installed the preflight passes and the wrapper reaches the idle gate, whose (fake)
    interpreter fails, so it refuses there: still nothing written and nothing launched."""
    import re

    w, r, log = _wrapper_repo(tmp_path, case)
    assert r.returncode == 1, r.stderr
    assert re.search(f"REFUSED: ({refusal})", r.stderr), r.stderr
    if case == "ok":
        assert "scripts.GPU.alphazero.idle_gate wait" in log.read_text()
        assert "export-2" not in log.read_text()
    else:
        assert not log.exists()
    assert sorted(p.name for p in (w / XM.OUTER2).iterdir()) == ["01_wrapper.sh"]
    assert not (w / XM.EV2).exists()
    assert (w / GATE_REL).read_text().endswith("HTB_MIX_EXPORT_AUTHORIZED = False\n")
    assert "GATE NOT RESTORED" not in r.stderr
