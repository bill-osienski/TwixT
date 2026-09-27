"""H5 manifest writer (implementation card §3, §5). NOT RUN FOR REAL.

The writer runs against SYNTHETIC qualification records committed in a temporary
git repository, their bound identity built by the real `h5_runner.bound_identity`
from the files as they are. PLACEHOLDER blocks stand in for step 6's (patched into
`SEED_BLOCKS` and ACCOUNTED for the test only); manifests land in temporary
directories; the fresh toolchain verification is stubbed. The three separate
refusals that keep an EARLY record from binding are each shown ALONE.
"""
import json
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_confirmatory_analysis as C5
from scripts.GPU.alphazero import h5_manifest_writer as W
from scripts.GPU.alphazero import h5_production_qualification as Q
from scripts.GPU.alphazero import h5_runner as H5
from tests.test_h4_pilot_feasibility import commit, fixture_leaks, git, new_repo
from tests.test_h4_runner import FAKE_TOOLCHAIN

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTENT = {k: FAKE_TOOLCHAIN[k] for k in R4.TOOLCHAIN_CONTENT}
PLACEHOLDERS = {k: (10_000 + 1_000 * k, 10_148 + 1_000 * k) for k in range(4)}


@pytest.fixture
def repo(tmp_path):
    return new_repo(tmp_path / "repo")


@pytest.fixture(autouse=True)
def placeholders(monkeypatch, tmp_path, repo):
    monkeypatch.setattr(H5, "SEED_BLOCKS", dict(PLACEHOLDERS))
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        tuple(REF.ACCOUNTED_SEED_INTERVALS) + tuple(PLACEHOLDERS.values()))
    monkeypatch.setattr(H5, "EVIDENCE_DIRS", {k: str(tmp_path / f"evidence-{k}")
                                              for k in range(4)})
    monkeypatch.setattr(Q, "BINDING_DIR", str(repo / "binding"))


def _ident():
    cfg = R4.frozen_argmax_config()
    return R4.frozen_incumbent_identity(cfg, design="H5_STUDY")


def fixture_record(repo, *, role="BINDING", where="binding", edit=None, result="CLEAN"):
    rec = {"record": "H5_PRODUCTION_QUALIFICATION", "role": role, "result": result,
           "checks": {"bound_identity": json.loads(json.dumps(
               H5.bound_identity(_ident(), CONTENT)))}}
    if edit:
        edit(rec)
    d = repo / where
    d.mkdir()
    (d / "record.json").write_text(json.dumps(rec))
    (d / "stages.jsonl").write_text("{}\n")
    commit(repo, d / "record.json", d / "stages.jsonl")
    return d / "record.json"


@pytest.fixture
def fresh(monkeypatch):
    box = {"jar_sha256": CONTENT["jar_sha256"], "jdk_components": CONTENT["jdk_components"]}
    monkeypatch.setattr(W, "_fresh_toolchain_content", lambda: dict(box))
    return box


def write(record, out):
    return W._write(str(record), str(out), _fixture=True)


def test_the_writer_builds_the_STUDY_manifest_from_the_registered_blocks(repo, fresh,
                                                                        tmp_path):
    path = write(fixture_record(repo), tmp_path / "manifests")
    m = json.loads(pathlib.Path(path).read_text())
    assert (m["stage"], len(m["segments"])) == ("H5_STUDY", 4)
    ident = _ident()
    for e in m["segments"]:
        k = e["segment"]
        lo, hi = PLACEHOLDERS[k]
        assert e["seeds"] == list(range(lo, hi))
        assert e["schedule"] == json.loads(json.dumps(H5.canonical_schedule(k, ident)))
        assert e["schedule"][0]["pair_id"] == f"h5s{k}-00"
        assert e["evidence_dir"] == H5.EVIDENCE_DIRS[k]
        head = H5.header_candidate(mode="study", segment=k, schedule=e["schedule"],
                                   incumbent_identity=ident, toolchain_content=CONTENT)
        assert R4.binding_differences(head, e, stage="H5_STUDY", with_toolchain=True) == [], \
            "the runner would bind to what the writer wrote"
    C5.load_manifest(path, _fixture=False)                  # the ANALYSIS reader accepts it
    assert m["qualification_record_sha256"] == F.sha256_file(str(repo / "binding" /
                                                                 "record.json"))


# ─────────────── the THREE refusals that keep an EARLY record from binding (§3) ───────────────

def test_refusal_1_an_EARLY_ROLE_is_refused_even_at_the_binding_path_with_a_matching_HEAD(
        repo, fresh, tmp_path):
    with pytest.raises(F.H4AnalysisRefused, match="an 'EARLY' record"):
        write(fixture_record(repo, role="EARLY"), tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


def test_refusal_2_a_BINDING_role_NOT_at_the_binding_path_is_refused(repo, fresh, tmp_path):
    with pytest.raises(F.H4AnalysisRefused, match="is not the BINDING record"):
        write(fixture_record(repo, where="early"), tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


def test_refusal_3_a_record_whose_CODE_predates_HEAD_is_refused(repo, fresh, tmp_path):
    """What an EARLY record looks like after step 6: its bound `code` is not HEAD's."""
    rec = fixture_record(repo, edit=lambda r: r["checks"]["bound_identity"]["code"].update(
        {H5.CODE[0]: "0" * 64}))
    with pytest.raises(F.H4AnalysisRefused, match="`code` at HEAD differs"):
        write(rec, tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


@pytest.mark.parametrize("edit,match", [
    (lambda r: r.update(result="STOP"), "not a CLEAN"),
    (lambda r: r.update(record="H4_PRODUCTION_QUALIFICATION"), "not a CLEAN H5"),
    (lambda r: r["checks"]["bound_identity"]["cards"].update({H5.CARDS[0]: "0" * 64}),
     "`cards` at HEAD differs"),
    (lambda r: r["checks"]["bound_identity"]["incumbent_identity"].update(design="H4_STUDY"),
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


def test_a_seed_the_REGISTRY_forbids_is_refused(repo, fresh, tmp_path, monkeypatch):
    lo, _hi = PLACEHOLDERS[2]
    monkeypatch.setattr(REF, "RETIRED_SEED_INTERVALS",
                        tuple(REF.RETIRED_SEED_INTERVALS) + ((lo, lo + 1),))
    with pytest.raises(F.H4AnalysisRefused, match="segment 2: seeds fail"):
        write(fixture_record(repo), tmp_path / "manifests")


def test_EMPTY_SEED_BLOCKS_are_refused(repo, fresh, tmp_path, monkeypatch):
    rec = fixture_record(repo)
    monkeypatch.setattr(H5, "SEED_BLOCKS", {})
    with pytest.raises(F.H4AnalysisRefused, match="no registered H5 block"):
        write(rec, tmp_path / "manifests")


def test_an_evidence_dir_that_ALREADY_EXISTS_is_refused(repo, fresh, tmp_path):
    pathlib.Path(H5.EVIDENCE_DIRS[1]).mkdir()
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
        W.write_manifest(str(fixture_record(repo)), str(tmp_path / "manifests"))
    assert not (tmp_path / "manifests").exists()


def test_the_PUBLIC_path_checks_EVERY_code_and_card_file_is_committed(repo, fresh, tmp_path,
                                                                    monkeypatch):
    seen = []
    monkeypatch.setattr(F, "check_durable",
                        lambda paths, *, _fixture: seen.append((list(paths), _fixture)))
    W.write_manifest(str(fixture_record(repo)), str(tmp_path / "manifests"))
    want = sorted(str(H5.REPO_ROOT / p) for p in (*H5.CODE, *H5.CARDS))
    assert any(sorted(p) == want and fx is False for p, fx in seen), \
        "the public path did not check that every code and card file is committed"


def test_the_CLI_refuses_as_a_fresh_subprocess(tmp_path):
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h5_manifest_writer",
                        "--record", str(tmp_path / "none.json"),
                        "--out-dir", str(tmp_path / "m")], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == W.EXIT_REFUSED, f"exit {r.returncode}, {r.stderr!r}"
    assert not (tmp_path / "m").exists()


def test_NO_public_entry_reaches_the_fixture_relaxation():
    assert fixture_leaks(W.__file__, {"write_manifest", "main"}) == []


def test_the_writer_is_OUTSIDE_code_and_the_play_path():
    assert "scripts/GPU/alphazero/h5_manifest_writer.py" not in H5.CODE
