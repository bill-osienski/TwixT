"""H4 proceed/stop artifact (analysis card §2, §2.1, §4, §4.1). SYNTHETIC FIXTURES ONLY.

Pilots are composed by `tests/test_h4_pilot_feasibility.py`'s builder from games
the real runner played in fixture mode; every file lives in a temporary git
repository. No JVM, no research seed, no real outcome.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import h4_pilot_decision as D
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from tests.test_h4_pilot_feasibility import (ROOT, _to_run_void, commit, feasibility, git,
                                             make_pilot, new_repo, templates)  # noqa: F401


@pytest.fixture
def repo(tmp_path):
    return new_repo(tmp_path / "repo")


def pilot_with_report(repo, templates, *, commit_report=True, **kw):
    man, res = make_pilot(repo, templates, **kw)
    rep = res.parent / "feasibility.json"
    feasibility(man, res, rep)
    if commit_report:
        commit(repo, rep)
    return man, res, rep


def decide(man, res, rep, out=None):
    out = out or pathlib.Path(res).parent / "decision.json"
    return D._decide(str(man), str(res), str(rep), str(out), _fixture=True)


def cli(man, res, rep, out):
    return D._main(["--manifest", str(man), "--results", str(res), "--report", str(rep),
                    "--out", str(out)], _fixture=True)


# ─────────────────────── 1. the decision is COMPUTED, and exits (§2, §4.1) ───────────────────────

def test_a_PROCEED_artifact_carries_the_frozen_fields_and_hashes(repo, templates):
    man, res, rep = pilot_with_report(repo, templates)
    out = res.parent / "decision.json"
    assert cli(man, res, rep, out) == 0
    art = json.loads(out.read_text())
    assert set(art) == {"decision", "rules_fired", "pilot_results_sha256",
                        "feasibility_report_sha256", "cards", "written_at"}
    assert (art["decision"], art["rules_fired"]) == ("PROCEED", [])
    assert art["pilot_results_sha256"] == F.sha256_file(str(res)), \
        "the artifact must carry the pilot results' sha256"
    assert art["feasibility_report_sha256"] == F.sha256_file(str(rep))
    assert sorted(art["cards"]) == sorted([F.CARD, F.RUNNER_CARD, F.REPLACEMENT_CARD])
    assert all(v == F.sha256_file(str(ROOT / c)) for c, v in art["cards"].items())


@pytest.mark.parametrize("kw,decision,code", [
    ({"total_s": 2391.0}, "STOP_RUNTIME", 2),
    ({"picks": [("red2", "black")] + [("cap", "black")] * 4 + [("red", "black")] * 11},
     "STOP_CAP", 2),
    ({"picks": [("red", "black")] * 16}, "STOP_COLLAPSE", 2),
    ({"edit": _to_run_void}, "VOID", 3)])
def test_STOP_and_VOID_decisions_and_their_exit_codes(repo, templates, kw, decision, code):
    man, res, rep = pilot_with_report(repo, templates, **kw)
    out = res.parent / "decision.json"
    assert cli(man, res, rep, out) == code
    assert json.loads(out.read_text())["decision"] == decision


def test_a_report_EDITED_to_say_PROCEED_is_refused_the_decision_is_never_typed(repo,
                                                                            templates):
    man, res, rep = pilot_with_report(repo, templates, commit_report=False,
                                      total_s=2391.0)
    stored = json.loads(rep.read_text())
    rep.write_text(json.dumps(dict(stored, verdict="PROCEED", rules_fired=[])))
    commit(repo, rep)                                         # durable, and still wrong
    with pytest.raises(F.H4AnalysisRefused, match="does not re-verify"):
        decide(man, res, rep)
    assert not (res.parent / "decision.json").exists()


def test_a_report_from_ANOTHER_PILOT_is_refused(repo, templates):
    man, res, _rep = pilot_with_report(repo, templates)
    _m2, _r2, rep2 = pilot_with_report(repo, templates, name="other",
                                       picks=[("red", "black")] * 16)
    with pytest.raises(F.H4AnalysisRefused, match="does not re-verify"):
        decide(man, res, rep2)


# ─────────────────────────────── 2. durability (§2.1) ───────────────────────────────

@pytest.mark.parametrize("which", ["manifest", "results", "report"])
@pytest.mark.parametrize("how", ["untracked", "modified"])
def test_each_file_the_decision_rests_on_must_be_COMMITTED_and_UNMODIFIED(
        repo, templates, which, how):
    man, res = make_pilot(repo, templates, to_commit=None)
    rep = res.parent / "feasibility.json"
    files = {"manifest": man, "results": res, "report": rep}
    commit(repo, man, res)
    feasibility(man, res, rep)
    commit(repo, rep)
    target = files[which]
    if how == "untracked":
        git(repo, "rm", "-q", "--cached", "--", str(target))
        commit(repo)                                          # records the removal
    else:
        target.write_text(target.read_text() + "\n")
    with pytest.raises(F.H4AnalysisRefused, match="not durable"):
        decide(man, res, rep)
    assert not (res.parent / "decision.json").exists()


# ─────────────────────────────── 3. output semantics (§2, §4.1) ───────────────────────────────

def test_an_OCCUPIED_artifact_is_refused_before_any_input_is_read(tmp_path):
    out = tmp_path / "decision.json"
    out.write_text("prior")
    with pytest.raises(F.H4AnalysisRefused, match="occupied"):
        D._decide(str(tmp_path / "m"), str(tmp_path / "r"), str(tmp_path / "f"), str(out),
                  _fixture=True)
    assert out.read_text() == "prior"


def test_the_artifact_must_go_into_the_PILOTS_evidence_directory(repo, templates, tmp_path):
    man, res, rep = pilot_with_report(repo, templates)
    with pytest.raises(F.H4AnalysisRefused, match="evidence directory"):
        decide(man, res, rep, out=tmp_path / "elsewhere.json")


def test_a_refusal_leaves_no_output_and_no_temporary_file(repo, templates):
    man, res, rep = pilot_with_report(repo, templates)
    res.write_text(res.read_text() + "\n")
    before = sorted(os.listdir(res.parent))
    with pytest.raises(F.H4AnalysisRefused):
        decide(man, res, rep)
    assert sorted(os.listdir(res.parent)) == before


# ───────────────── 4. the fixture relaxation is unreachable (§4) ─────────────────

def test_the_PUBLIC_entry_refuses_a_committed_FIXTURE(repo, templates):
    man, res, rep = pilot_with_report(repo, templates)
    with pytest.raises(F.H4AnalysisRefused, match="does not resolve into this repository"):
        D.write_decision(str(man), str(res), str(rep), str(res.parent / "d.json"))
    assert not (res.parent / "d.json").exists()


def test_the_PUBLIC_entry_refuses_FIXTURE_CONTENT_even_inside_the_repository(
        repo, templates, monkeypatch):
    man, res, rep = pilot_with_report(repo, templates)
    monkeypatch.setattr(F.R, "REPO_ROOT", repo.resolve())
    with pytest.raises(F.H4AnalysisRefused, match="negative seed"):
        D.write_decision(str(man), str(res), str(rep), str(res.parent / "d.json"))
    assert not (res.parent / "d.json").exists()


def test_the_CLI_refuses_a_committed_FIXTURE_as_a_fresh_subprocess(repo, templates):
    man, res, rep = pilot_with_report(repo, templates)
    out = res.parent / "d.json"
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h4_pilot_decision",
                        "--manifest", str(man), "--results", str(res), "--report", str(rep),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == D.EXIT_REFUSED, f"exit {r.returncode}, stderr {r.stderr!r}"
    assert not out.exists()


def test_NO_public_entry_of_the_decision_can_reach_the_fixture_relaxation():
    from tests.test_h4_pilot_feasibility import fixture_leaks
    assert fixture_leaks(D.__file__, {"write_decision", "main"}) == []
