"""The planted qualification's outer wrapper (card §9 step 2): `docs/superpowers/evidence/
policy-head-capacity-planted-run/01_wrapper.sh`. Each case runs the REAL wrapper (or, for the ceiling, a copy whose
only diff is the two ceiling lines) inside a THROWAWAY git repo with the real layout: a stdlib stub in place of the
module, the gate opened by a one-line commit, `.venv/bin/python` linked to this interpreter and an EMPTY file where
the checkpoint goes (never read). No checkpoint, no real run; this repo's gate is never touched."""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import policy_head_capacity_planted as PL
from scripts.GPU.alphazero import policy_head_capacity_planted_authorization as GATE

ROOT = pathlib.Path(__file__).resolve().parents[1]
WRAPPER_REL = "docs/superpowers/evidence/policy-head-capacity-planted-run/01_wrapper.sh"
GATE_REL = "scripts/GPU/alphazero/policy_head_capacity_planted_authorization.py"
STUB_REL = "scripts/GPU/alphazero/policy_head_capacity_planted.py"
NAME = "POLICY_HEAD_CAPACITY_PLANTED_AUTHORIZED"
WRAPPER = (ROOT / WRAPPER_REL).read_text()

STUB = '''"""TEST STUB for policy_head_capacity_planted inside a throwaway repo; behaviour from STUB_MODE."""
import os, pathlib, signal, sys, time
GATE = pathlib.Path("scripts/GPU/alphazero/policy_head_capacity_planted_authorization.py")
mode = os.environ["STUB_MODE"]
print("ARGV", sys.argv[1:])
print("GATE_OPEN", "AUTHORIZED = True" in GATE.read_text(), flush=True)
if mode == "done":
    pathlib.Path("docs/superpowers/evidence/policy-head-capacity-planted").mkdir()
    sys.exit(0)
if mode == "void":
    sys.exit(3)
if mode == "refused":
    print("REFUSED: stub", file=sys.stderr)
    sys.exit(4)
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode in ("hang", "stubborn"):
    time.sleep(120)
if mode == "corrupt_gate":
    GATE.write_text(GATE.read_text().replace("AUTHORIZED = True", "AUTHORIZED = 1"))
    sys.exit(0)
sys.exit(9)
'''


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


def _repo(tmp_path, wrapper_text=WRAPPER, open_gate=True):
    repo = tmp_path / "repo"
    files = {WRAPPER_REL: wrapper_text, GATE_REL: (ROOT / GATE_REL).read_text(), STUB_REL: STUB,
             ".gitignore": ".venv/\ncheckpoints/\n", "README": "x\n"}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / WRAPPER_REL).chmod(0o755)
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    ckpt = repo / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "closed")
    if open_gate:
        g = repo / GATE_REL
        g.write_text(g.read_text().replace(f"{NAME} = False", f"{NAME} = True"))
        _git(repo, "commit", "-q", "-am", "open the gate")
    else:
        (repo / "README").write_text("y\n")
        _git(repo, "commit", "-q", "-am", "not a gate opening")
    return repo


def _run(repo, mode, timeout=60):
    env = {**os.environ, "STUB_MODE": mode}
    p = subprocess.run(["/bin/bash", str(repo / WRAPPER_REL)], cwd="/", env=env, capture_output=True, text=True,
                       timeout=timeout)
    out = repo / pathlib.Path(WRAPPER_REL).parent
    outer = json.loads((out / "outer.json").read_text()) if (out / "outer.json").exists() else None
    return p, outer, out


def _restored(repo):
    """The gate file is byte-identical to the closed commit's, and closed."""
    closed = subprocess.run(["git", "show", f"HEAD~1:{GATE_REL}"], cwd=repo, capture_output=True, text=True).stdout
    text = (repo / GATE_REL).read_text()
    return text == closed and f"{NAME} = False" in text


def _ceiling_copy(term_s=2, grace_s=1):
    text = WRAPPER.replace("TERM_S=890\n", f"TERM_S={term_s}\n").replace("GRACE_S=10\n", f"GRACE_S={grace_s}\n")
    diff = [(a, b) for a, b in zip(WRAPPER.splitlines(), text.splitlines()) if a != b]
    assert diff == [("TERM_S=890", f"TERM_S={term_s}"), ("GRACE_S=10", f"GRACE_S={grace_s}")]
    return text


# ─────────────────────────────── the committed wrapper, statically ───────────────────────────────

def test_the_wrapper_names_this_gate_this_module_and_the_card_s_ceiling():
    assert os.access(ROOT / WRAPPER_REL, os.X_OK)
    assert f"GATE={NAME}\n" in WRAPPER and hasattr(GATE, NAME) and getattr(GATE, NAME) is False
    assert f"GATE_REL={GATE_REL}\n" in WRAPPER and (ROOT / GATE_REL).is_file()
    assert "MODULE=scripts.GPU.alphazero.policy_head_capacity_planted\n" in WRAPPER
    assert f"RUN_REL={PL.EVIDENCE_DIR}\n" in WRAPPER and f"OUT_REL={PL.EVIDENCE_DIR}-run\n" in WRAPPER
    term, grace = (int(re.search(rf"^{k}=(\d+)$", WRAPPER, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (890, 10) and term + grace == PL.CAP_S
    assert "timeout -k" not in WRAPPER and 'run --checkpoint "$CKPT"' in WRAPPER
    assert subprocess.run(["/bin/bash", "-n", str(ROOT / WRAPPER_REL)]).returncode == 0


# ─────────────────────────────── each exit, and the gate after each ───────────────────────────────

def test_A_CLEAN_EXIT_maps_to_COMPLETED_and_restores_the_gate(tmp_path):
    repo = _repo(tmp_path)
    p, outer, out = _run(repo, "done")
    assert p.returncode == 0, p.stderr
    assert outer["inner_exit"] == 0 and outer["outcome"] == "COMPLETED" and outer["outer_exit"] == 0
    assert outer["gate_restored"] is True and _restored(repo) and outer["survivors"] == ""
    stdout = (out / "run.stdout.txt").read_text()
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert f"ARGV ['run', '--checkpoint', '{ckpt}']" in stdout and "GATE_OPEN True" in stdout   # open DURING the run
    assert outer["pre_open"] == subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=repo, capture_output=True,
                                               text=True).stdout.strip()


@pytest.mark.parametrize("mode, code, outcome", [("void", 3, "VOID"), ("refused", 4, "REFUSED")])
def test_A_VOID_AND_A_REFUSED_RUN_map_through_and_restore_the_gate(tmp_path, mode, code, outcome):
    repo = _repo(tmp_path)
    p, outer, out = _run(repo, mode)
    assert p.returncode == code and outer["inner_exit"] == code and outer["outcome"] == outcome
    assert outer["gate_restored"] is True and _restored(repo)
    if mode == "refused":
        assert "REFUSED: stub" in (out / "run.stderr.txt").read_text()


def test_THE_WRAPPER_REFUSES_a_head_that_does_not_open_the_gate(tmp_path):
    repo = _repo(tmp_path, open_gate=False)
    p, outer, out = _run(repo, "done")
    assert p.returncode == 1 and "REFUSED: HEAD does not change only" in p.stderr
    assert outer is None and not (out / "run.stdout.txt").exists() and _restored(repo)


def test_THE_WRAPPER_REFUSES_a_dirty_tree_and_still_closes_the_open_gate(tmp_path):
    repo = _repo(tmp_path)
    (repo / "stray.txt").write_text("x")
    assert f"{NAME} = True" in (repo / GATE_REL).read_text()
    p, outer, out = _run(repo, "done")
    assert p.returncode == 1 and "REFUSED: the tree is not clean" in p.stderr
    assert outer is None and not (out / "run.stdout.txt").exists() and _restored(repo)


def test_THE_CEILING_TERM_fires_maps_to_TIMEOUT_and_restores_the_gate(tmp_path):
    repo = _repo(tmp_path, wrapper_text=_ceiling_copy())
    p, outer, _ = _run(repo, "hang")
    assert p.returncode == 124 and outer["inner_exit"] == 124 and outer["outcome"] == "TIMEOUT"
    assert outer["wall_seconds"] < 10 and outer["survivors"] == ""
    assert outer["gate_restored"] is True and _restored(repo)


def test_THE_CEILING_KILL_after_an_ignored_TERM_and_the_gate_is_restored(tmp_path):
    repo = _repo(tmp_path, wrapper_text=_ceiling_copy())
    p, outer, _ = _run(repo, "stubborn")
    assert p.returncode == 137 and outer["inner_exit"] == 137 and outer["outcome"] == "TIMEOUT"
    assert outer["wall_seconds"] < 10 and outer["survivors"] == ""
    assert outer["gate_restored"] is True and _restored(repo)


def test_THE_READ_BACK_IS_NOT_VACUOUS_an_unrestorable_gate_exits_5(tmp_path):
    repo = _repo(tmp_path)
    p, outer, _ = _run(repo, "corrupt_gate")
    assert p.returncode == 5 and outer["inner_exit"] == 0 and outer["outer_exit"] == 5
    assert outer["gate_restored"] is False and "GATE NOT RESTORED" in p.stderr and not _restored(repo)


def test_the_outputs_are_create_only(tmp_path):
    repo = _repo(tmp_path)
    _run(repo, "void")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "evidence and the restored gate")
    g = repo / GATE_REL                                     # a fresh one-line opening, clean tree: only the
    g.write_text(g.read_text().replace(f"{NAME} = False", f"{NAME} = True"))   # existing outputs can refuse
    _git(repo, "commit", "-q", "-am", "open the gate again")
    p, _, out = _run(repo, "done")
    assert p.returncode == 1 and "run.stdout.txt exists (one-shot)" in p.stderr and _restored(repo)
    assert (out / "run.stdout.txt").read_text().count("ARGV") == 1          # the stub did not run a second time
