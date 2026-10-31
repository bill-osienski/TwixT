"""The combination policy pilot's CPU TRAINING (card `docs/superpowers/2026-10-28-combination-policy-pilot-card.md`
§5 step 4), built only: the wrapper (`docs/superpowers/evidence/combination-policy-train-run/01_wrapper.sh`) with the
ceiling the user set before approval (ledger, decision note after entry 43: 2,820 s). CPU only; nothing here loads a
model, touches a device or opens a real gate.

The wrapper is corpus generation's wrapper with only the stated differences, asserted line by line. The ceiling is
re-derived here from the committed timing record and the committed corpus counts, never copied from the ledger. THE
FAKE-LAUNCH CHECKS run the REAL training wrapper's bytes in a throwaway git repository with the throughput tests' stubs
(tests/test_combination_policy_launch) moved to training's paths: the refusals, the exact command and the ceiling."""
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import combination_policy_pilot as P
from tests.test_combination_policy_launch import STUB_BG, STUB_IDLE, STUB_PILOT, _alive, _git

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
WRAPPER_GEN = EV / "combination-policy-generate-run/01_wrapper.sh"
WRAPPER = EV / "combination-policy-train-run/01_wrapper.sh"
GATE_NAME = "COMBINATION_POLICY_TRAIN_AUTHORIZED"
TIMING_COMMIT = "c0bf2452"          # the timing record's own head (record.json "head"): it is kept as written


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """EVERY test (#74): the pilot's real seams, its record boundary and the real cache are unreachable."""
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    monkeypatch.setattr(P, "_real_gen_seams", unreachable)
    monkeypatch.setattr(P, "_real_train_seams", unreachable)
    monkeypatch.setattr(P, "conclude", unreachable)
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(P.SETTING, P.SETTING_VALUE)


def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


def _ceiling():
    """The decision note's derivation, from the committed records: the timing record's formula on the corpus's real
    counts (rows from games.jsonl's row counts; held-out by the recipe's slot rule), x the card's 1.5, rounded up to
    the minute. games.jsonl is read for slots and row counts only -- never a winner or reason."""
    timing = json.loads((ROOT / P.TIMING_EVIDENCE / "record.json").read_text(encoding="utf-8"))
    games = [json.loads(line) for line in
             (ROOT / P.KINDS["generate"]["evidence"] / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = [(g["slot"], g["n_rows"]) for g in games]
    rows = sum(n for _s, n in counts)
    held = sum(n for s, n in counts if s % P.RECIPE["heldout_mod"] == P.RECIPE["heldout_rem"])
    t = timing["timing"]
    projected = math.fsum([t["mean_update_seconds"] * P.RECIPE["updates"],
                           t["per_row_pair_forward_seconds"] * (rows + held),
                           t["per_row_pair_forward_seconds"] * held,
                           t["per_to_tensor_seconds"] * rows])
    return rows, held, projected, int(math.ceil(timing["spec"]["ceiling_factor"] * projected / 60.0) * 60), timing


def test_TRAINING_IS_BUILT_NOT_RUN_its_gate_closed_and_the_timing_record_kept_as_written():
    """BUILT 2026-10-31: the gate is closed; the run directory holds the wrapper only; no training evidence exists.
    Training is admitted through the REAL committed readers (generation's record and outer.json, and the timing). The
    timing record still says 2,700 s, unchanged since it was committed: the 2,820 s ceiling lives in the wrapper and
    the ledger, not in an edited record. Read-only."""
    from scripts.GPU.alphazero import combination_policy_train_authorization as G
    assert getattr(G, GATE_NAME) is False
    assert sorted(x.name for x in WRAPPER.parent.iterdir()) == ["01_wrapper.sh"]
    assert not (ROOT / P.KINDS["train"]["evidence"]).exists()
    assert P.admitted("train", P._read_record, P._read_outer) == {
        P.KINDS["generate"]["evidence"]: "COMPLETED", P.TIMING_EVIDENCE: "COMPLETED",
        f"{pathlib.Path(P.KINDS['generate']['plan']).parent}/outer.json": "COMPLETED"}
    rows, held, _projected, _ceiling_s, timing = _ceiling()
    assert timing["projection"]["ceiling_seconds"] == 2700 and timing["head"].startswith(TIMING_COMMIT)
    assert (timing["spec"]["projected_rows"], timing["spec"]["projected_heldout_rows"]) == (37000, 3700)
    assert (rows, held) != (37000, 3700)
    assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "56e6f386", "HEAD", "--", P.TIMING_EVIDENCE,
                           P.KINDS["generate"]["evidence"],
                           str(pathlib.Path(P.KINDS["generate"]["plan"]).parent)]).returncode == 0


# ─────────────────────────────── the wrapper's bytes ───────────────────────────────

def test_THE_WRAPPER_IS_GENERATION_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The paths, the gate, `run train` and TERM_S, line for line; everything else byte-identical. Its header is
    training's paragraph followed by generation's header unchanged."""
    gen, train = _code(WRAPPER_GEN).splitlines(), _code(WRAPPER).splitlines()
    assert len(gen) == len(train)
    changed = [(a, b) for a, b in zip(gen, train) if a != b]
    assert [b for _a, b in changed] == [
        "OUT_REL=docs/superpowers/evidence/combination-policy-train-run",
        "RUN_REL=docs/superpowers/evidence/combination-policy-train",
        "GATE_REL=scripts/GPU/alphazero/combination_policy_train_authorization.py",
        f"GATE={GATE_NAME}", "TERM_S=2810",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        '"$PY" -B -m "$MODULE" run train --checkpoint "$CKPT" \\'], changed
    head_gen = WRAPPER_GEN.read_text().split("set -u\n", 1)[0]
    head = WRAPPER.read_text().split("set -u\n", 1)[0]
    assert head.startswith("#!/bin/bash\n# THE COMBINATION POLICY PILOT: CPU TRAINING, THEN THE CHECK (CARD STEP 4)")
    assert head.endswith(head_gen[len("#!/bin/bash\n"):])
    ours = head[:-len(head_gen) + len("#!/bin/bash\n")]
    assert "decision note after entry 43" in ours and "2,820 s" in ours and "2830 s" in ours
    assert os.access(WRAPPER, os.R_OK) and subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0


def test_THE_CEILING_IS_1_5_X_THE_TIMING_FORMULA_ON_THE_REAL_CORPUS_rounded_up_2820_S():
    """TERM at 2,810 s, KILL at 2,820 s, the observation through 2,830 s; 2,820 = the card's rule on the real counts."""
    rows, held, projected, ceiling_s, _timing = _ceiling()
    assert (rows, held) == (44747, 4158) and round(projected, 1) == 1865.3 and ceiling_s == 2820
    w = _code(WRAPPER)
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (2810, 10) and term + grace == ceiling_s and term + 2 * grace == 2830


def test_THE_WRAPPER_SERVES_THE_PILOT_S_TRAIN_PATHS():
    w = _code(WRAPPER)
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    kind = P.KINDS["train"]
    assert run == kind["evidence"] and out == str(WRAPPER.parent.relative_to(ROOT)) == f"{kind['evidence']}-run"
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == \
        str(pathlib.Path(kind["gate"][0].__file__).relative_to(ROOT))
    assert kind["gate"][1] == GATE_NAME and f"RECORD_KIND={P.RECORD_KIND}\n" in w
    others = {p for k in ("throughput", "throughput2", "generate")
              for p in (P.KINDS[k]["evidence"], str(pathlib.Path(P.KINDS[k]["plan"]).parent))}
    assert {run, out} & others == set() and P.TIMING_EVIDENCE not in (run, out)
    assert "CKPT=\"$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors\"" in w


# ─────────────────────────────── fake launches of the REAL training wrapper ───────────────────────────────

#: the throughput tests' stub at training's paths
STUB_PILOT_TRAIN = STUB_PILOT.replace("combination-policy-throughput", "combination-policy-train")
assert STUB_PILOT_TRAIN.count("combination-policy-train") == 2


def _fake_repo(tmp, term_s=None):
    """A throwaway repository: the real training wrapper (TERM_S changed only when a test says so), its gate opened
    by a one-line commit, and three stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / "combination_policy_train_authorization.py"
    gate.write_text(f'"""stub gate"""\n{GATE_NAME} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "combination_policy_pilot.py").write_text(STUB_PILOT_TRAIN)
    wdir = repo / "docs/superpowers/evidence/combination-policy-train-run"
    wdir.mkdir(parents=True)
    raw = WRAPPER.read_bytes()
    if term_s is not None:
        assert raw.count(b"\nTERM_S=2810\n") == 1
        raw = raw.replace(b"\nTERM_S=2810\n", f"\nTERM_S={term_s}\n".encode())
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
    gate.write_text(gate.read_text().replace(f"{GATE_NAME} = False", f"{GATE_NAME} = True"))
    _git(repo, "commit", "-qam", "open the gate")
    (repo / "scratch").mkdir()
    return repo, wdir, gate


def _launch(repo, wdir, idle="idle", bg="clear", run="ok", timeout=60):
    env = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "PYTHONPATH")}
    env.update(FAKE_IDLE=idle, FAKE_BG=bg, FAKE_RUN=run, FAKE_OUT=str(repo / "scratch"), PYTHONDONTWRITEBYTECODE="1",
               FAKE_CARD_SHA=P.CARD_SHA256)
    return subprocess.run(["bash", str(wdir / "01_wrapper.sh")], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=timeout)


@pytest.mark.parametrize("idle, bg", [("busy", "clear"), ("idle", "boinc"), ("idle", "failed")],
                         ids=["busy", "boinc", "check-fails"])
def test_FAKE_LAUNCH_A_BUSY_MACHINE_BOINC_OR_A_FAILED_CHECK_PREVENTS_THE_LAUNCH(tmp_path, idle, bg):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, idle, bg)
    assert p.returncode == 1 and ": nothing launched" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]
    assert not (repo / "docs/superpowers/evidence/combination-policy-train").exists()
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_A_GATE_OPENED_WITH_ANY_OTHER_CHANGE_IS_REFUSED(tmp_path):
    """The one-line opening is the only accepted HEAD: an opening commit that also touches another file refuses."""
    repo, wdir, gate = _fake_repo(tmp_path)
    _git(repo, "reset", "-q", "--soft", "HEAD~1")
    (repo / "scripts/GPU/alphazero/idle_gate.py").write_text(STUB_IDLE + "\n")
    _git(repo, "commit", "-qam", "open the gate and change something else")
    p = _launch(repo, wdir)
    assert p.returncode == 1 and "does not change only" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr and f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_RUN_TRAIN_with_its_exact_arguments_and_ceiling(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", "train", "--checkpoint", ckpt] and launched["tf32"] == "0"
    assert (wdir / "pgid.txt").read_text() == f"{launched['pgid']}\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == 2810 and outer["grace_s"] == 10 and outer["record_by"] == "script"
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-train/record.json").read_text())
    assert not P.outer_faults(outer, rec)
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_at_training_s_paths(tmp_path):
    """TERM_S changed to 2 in this copy only: the run and its child outlive it; the group TERM ends both, and the
    wrapper's VOID record lands in training's evidence directory."""
    repo, wdir, gate = _fake_repo(tmp_path, term_s=2)
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
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-train/record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
    assert f"{GATE_NAME} = False\n" in gate.read_text()
