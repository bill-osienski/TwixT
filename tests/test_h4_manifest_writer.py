"""H4 step-4d manifest writer, built in 4b (step-4 card §4). NOT RUN FOR REAL.

The writer runs against a SYNTHETIC qualification record, committed in a
temporary git repository, whose bound identity is built by the real
`bound_identity` from the files as they are. The five blocks and the shared
registry are the REAL ones -- the manifests land in temporary directories, and
no seed reaches an incumbent. The fresh toolchain verification is stubbed (the
toolchain lives outside the repository); the fixture relaxation (`_fixture`)
reaches only the private core.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import h4_manifest_writer as W
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_runner as R
from tests.test_h4_pilot_feasibility import commit, fixture_leaks, git, new_repo
from tests.test_h4_runner import FAKE_TOOLCHAIN

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTENT = {k: FAKE_TOOLCHAIN[k] for k in R.TOOLCHAIN_CONTENT}


def fixture_record(repo, edit=None, result="CLEAN"):
    cfg = R.frozen_argmax_config()
    ident = R.frozen_incumbent_identity(cfg, design="H4_PILOT")
    rec = {"record": "H4_PRODUCTION_QUALIFICATION", "result": result,
           "checks": {"bound_identity": json.loads(json.dumps(R.bound_identity(ident, CONTENT)))}}
    if edit:
        edit(rec)
    d = repo / "requal"
    d.mkdir()
    (d / "record.json").write_text(json.dumps(rec))
    (d / "stages.jsonl").write_text("{}\n")
    commit(repo, d / "record.json", d / "stages.jsonl")
    return d / "record.json"


@pytest.fixture
def repo(tmp_path):
    return new_repo(tmp_path / "repo")


@pytest.fixture
def fresh(monkeypatch):
    box = {"jar_sha256": CONTENT["jar_sha256"], "jdk_components": CONTENT["jdk_components"]}
    monkeypatch.setattr(W, "_fresh_toolchain_content", lambda: dict(box))
    return box


def write(record, out):
    return W._write(str(record), str(out), _fixture=True)


def test_the_writer_builds_BOTH_manifests_from_the_registered_blocks(repo, fresh, tmp_path):
    paths = write(fixture_record(repo), tmp_path / "manifests")
    pilot, study = (json.loads(pathlib.Path(p).read_text()) for p in paths)
    assert (pilot["stage"], len(pilot["segments"])) == ("H4_PILOT", 1)
    assert (study["stage"], len(study["segments"])) == ("H4_STUDY", 4)
    cfg = R.frozen_argmax_config()
    for mode, m in (("pilot", pilot), ("study", study)):
        ident = R.frozen_incumbent_identity(cfg, design=R.DESIGNS[mode])
        for e in m["segments"]:
            seg = e["segment"]
            lo, hi = R.SEED_BLOCKS[(mode, seg)]
            assert e["seeds"] == list(range(lo, hi))
            assert e["schedule"] == json.loads(json.dumps(R.canonical_schedule(mode, seg, ident)))
            assert e["evidence_dir"] == R.EVIDENCE_DIRS[(mode, seg)]
            head = R.header_candidate(mode=mode, segment=seg, schedule=e["schedule"],
                                      incumbent_identity=ident, toolchain_content=CONTENT)
            assert R.binding_differences(head, e, stage=R.DESIGNS[mode],
                                         with_toolchain=True) == [], \
                "the runner would bind to what the writer wrote"
    for p, stage in zip(paths, (F.PILOT, F.STUDY)):
        F.load_manifest(p, stage, _fixture=False)          # the ANALYSIS reader accepts it
    assert pilot["qualification_record_sha256"] == F.sha256_file(
        str(repo / "requal" / "record.json"))


@pytest.mark.parametrize("edit,match", [
    (lambda r: r.update(result="STOP"), "not a CLEAN"),
    (lambda r: r["checks"]["bound_identity"]["code"].update({R.CODE[0]: "0" * 64}),
     "`code` at HEAD differs"),
    (lambda r: r["checks"]["bound_identity"]["cards"].update({R.CARDS[4]: "0" * 64}),
     "`cards` at HEAD differs"),
    (lambda r: r["checks"]["bound_identity"]["incumbent_identity"].update(design="X"),
     "`incumbent_identity` at HEAD differs"),
    (lambda r: r["checks"]["bound_identity"]["t1j_runtime"].update(ply_cap=274),
     "`t1j_runtime` at HEAD differs")])
def test_a_record_that_no_longer_BINDS_is_refused_and_nothing_written(repo, fresh, tmp_path,
                                                                     edit, match):
    with pytest.raises(F.H4AnalysisRefused, match=match):
        write(fixture_record(repo, edit=edit), tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


@pytest.mark.parametrize("field", ["jar_sha256", "jdk_components"])
def test_a_toolchain_that_verifies_differently_NOW_is_refused(repo, fresh, tmp_path, field):
    fresh[field] = "changed"
    with pytest.raises(F.H4AnalysisRefused, match=f"`{field}` differs"):
        write(fixture_record(repo), tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


def test_a_seed_the_REGISTRY_forbids_is_refused(repo, fresh, tmp_path, monkeypatch):
    lo, _hi = R.SEED_BLOCKS[("study", 2)]
    monkeypatch.setattr(REF, "EXPOSED_SEED_INTERVALS",
                        tuple(REF.EXPOSED_SEED_INTERVALS) + ((lo, lo + 1),))
    with pytest.raises(F.H4AnalysisRefused, match="study segment 2: seeds fail"):
        write(fixture_record(repo), tmp_path / "manifests")


def test_an_evidence_dir_that_ALREADY_EXISTS_is_refused(repo, fresh, tmp_path, monkeypatch):
    (tmp_path / "taken").mkdir()
    monkeypatch.setattr(R, "EVIDENCE_DIRS", {**R.EVIDENCE_DIRS,
                                             ("study", 1): str(tmp_path / "taken")})
    with pytest.raises(F.H4AnalysisRefused, match="already exists"):
        write(fixture_record(repo), tmp_path / "manifests")


def test_an_UNTRACKED_record_is_refused(repo, fresh, tmp_path):
    rec = fixture_record(repo)
    git(repo, "rm", "-q", "--cached", "--", str(rec))
    commit(repo)
    with pytest.raises(F.H4AnalysisRefused, match="not durable"):
        write(rec, tmp_path / "manifests")


def test_an_OCCUPIED_destination_is_refused_before_anything(tmp_path):
    (tmp_path / "manifests").mkdir()
    with pytest.raises(F.H4AnalysisRefused, match="occupied"):
        write(tmp_path / "no-record.json", tmp_path / "manifests")


def test_the_PUBLIC_entry_refuses_a_record_outside_THIS_repository(repo, fresh, tmp_path):
    with pytest.raises(F.H4AnalysisRefused, match="does not resolve into this repository"):
        W.write_manifests(str(fixture_record(repo)), str(tmp_path / "manifests"))
    assert not (tmp_path / "manifests").exists()


def test_the_PUBLIC_path_checks_EVERY_code_and_card_file_is_committed(repo, fresh, tmp_path,
                                                                    monkeypatch):
    """The production path's durability covers what the manifests hash."""
    seen = []
    monkeypatch.setattr(F, "check_durable",
                        lambda paths, *, _fixture: seen.append((list(paths), _fixture)))
    W.write_manifests(str(fixture_record(repo)), str(tmp_path / "manifests"))
    want = sorted(str(R.REPO_ROOT / p) for p in (*R.CODE, *R.CARDS))
    assert any(sorted(p) == want and fx is False for p, fx in seen), \
        "the public path did not check that every code and card file is committed"


def test_the_CLI_refuses_as_a_fresh_subprocess(tmp_path):
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h4_manifest_writer",
                        "--record", str(tmp_path / "none.json"),
                        "--out-dir", str(tmp_path / "m")], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == W.EXIT_REFUSED, f"exit {r.returncode}, {r.stderr!r}"
    assert not (tmp_path / "m").exists()


def test_NO_public_entry_reaches_the_fixture_relaxation():
    assert fixture_leaks(W.__file__, {"write_manifests", "main"}) == []


def test_the_writer_is_OUTSIDE_code_and_the_play_path():
    assert "scripts/GPU/alphazero/h4_manifest_writer.py" not in R.CODE
