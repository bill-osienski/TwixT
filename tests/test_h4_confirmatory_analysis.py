"""H4 confirmatory analysis (analysis card §3, §4, §4.1). SYNTHETIC FIXTURES ONLY.

Every study here is COMPOSED from games the real runner played in fixture mode
(`tests/test_h4_pilot_feasibility.py`), with negative seeds and
`H4_SYNTHETIC_FIXTURE` headers, inside a temporary git repository: 4 segments x
74 pairs, the pilot chain before it (manifest, results, feasibility report,
decision artifact) built by the real feasibility and decision cores. No JVM, no
research seed, no real outcome. The gate stays CLOSED in the source; tests that
need it open patch it for their own duration only.
"""
import ast
import builtins
import collections
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import h4_confirmatory_analysis as C
from scripts.GPU.alphazero import h4_pilot_decision as D
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_runner as R
from tests.test_h4_pilot_feasibility import (ROOT, commit, compose, edit_records, entry,
                                             feasibility, fixture_leaks, git, make_pilot,
                                             new_repo, templates)  # noqa: F401

ALL_CAP = [("cap", "cap")] * 74
#: pair scores 1, 0.5, 0, 0.75 -- incumbent points (A) + (B), halved
MIXED = [("red", "black"), ("cap", "cap"), ("black", "red"), ("red", "cap")]
SCORE = {("red", "black"): 1.0, ("cap", "cap"): 0.5, ("black", "red"): 0.0,
         ("red", "cap"): 0.75}
MODULES = {"feasibility": F, "decision": D, "confirmatory": C}


def build_study(root, templates, picks=ALL_CAP, *, pilot_kw=None):
    repo = new_repo(root)
    man, res = make_pilot(repo, templates, **(pilot_kw or {}))
    rep, art = res.parent / "feasibility.json", res.parent / "decision.json"
    feasibility(man, res, rep)
    commit(repo, rep)
    D._decide(str(man), str(res), str(rep), str(art), _fixture=True)
    sd = repo / "study"
    sd.mkdir()
    entries = [entry(k, 74, tag="s", seed0=1000 * (k + 1)) for k in range(4)]
    smani = sd / "manifest.json"
    smani.write_text(json.dumps({"stage": F.STUDY, "segments": entries}))
    segs = []
    for k, e in enumerate(entries):
        segs.append(sd / f"segment{k}.jsonl")
        compose(segs[-1], templates, e, [picks[j % len(picks)] for j in range(74)])
    commit(repo, art, smani, *segs)
    return {"repo": repo, "pilot_manifest": man, "study_manifest": smani, "artifact": art,
            "pilot_results": res, "feasibility_report": rep, "segments": segs,
            "entries": entries}


def args(study, **over):
    kw = {k: str(study[k]) for k in ("pilot_manifest", "study_manifest", "artifact",
                                     "pilot_results", "feasibility_report")}
    kw["segments"] = [str(s) for s in study["segments"]]
    kw["out"] = str(study["repo"] / "estimate.json")
    kw.update(over)
    return kw


def confirm(study, **over):
    return C._run(**args(study, **over), _fixture=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory, templates):
    return build_study(tmp_path_factory.mktemp("h4study") / "repo", templates)


def clone(study, tmp_path):
    """A private copy -- repository and all -- for a test that edits files."""
    dst = tmp_path / "repo"
    shutil.copytree(study["repo"], dst)
    move = lambda p: dst / pathlib.Path(p).relative_to(study["repo"])   # noqa: E731
    out = {k: move(v) for k, v in study.items()
           if k not in ("repo", "segments", "entries")}
    return {**out, "repo": dst, "segments": [move(s) for s in study["segments"]],
            "entries": study["entries"]}


@pytest.fixture
def gate_open(monkeypatch):
    monkeypatch.setattr(C, "H4_STUDY_AGGREGATION_AUTHORIZED", True)


# ─────────────────────────── 1. THE GATE comes first (§3.1) ───────────────────────────

def test_the_gate_is_false_as_published():
    assert C.H4_STUDY_AGGREGATION_AUTHORIZED is False


class Spy:
    """Every open / stat / hash / subprocess the call makes, by path."""

    def __init__(self, monkeypatch, root):
        self.seen, root = [], str(root)

        def wrap(owner, name):
            real = getattr(owner, name)

            def spy(*a, **k):
                p = a[0] if a else None
                if isinstance(p, (list, tuple)) or (isinstance(p, (str, os.PathLike))
                                                    and root in str(os.fspath(p))):
                    self.seen.append((name, str(p)))
                return real(*a, **k)
            monkeypatch.setattr(owner, name, spy)
        for owner, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"),
                            (os, "lstat"), (F, "sha256_file"), (subprocess, "run")):
            wrap(owner, name)


def test_a_CLOSED_gate_opens_stats_hashes_and_runs_NOTHING(study, monkeypatch):
    spy = Spy(monkeypatch, study["repo"])
    with pytest.raises(C.H4AggregationUnauthorized):
        C.run_confirmatory(**args(study))
    assert spy.seen == []
    assert not (study["repo"] / "estimate.json").exists()


def test_the_input_spy_is_NOT_VACUOUS(study, monkeypatch, gate_open):
    """CLEAN-BASELINE CONTROL: with the gate open the same spy sees the reads."""
    spy = Spy(monkeypatch, study["repo"])
    with pytest.raises(F.H4AnalysisRefused):
        C.run_confirmatory(**args(study))                     # public: refuses fixtures
    names = {n for n, _p in spy.seen}
    assert {"lstat", "run", "open"} <= names, names


def test_the_CLI_with_the_gate_closed_exits_5_before_touching_ANY_path(tmp_path):
    """Nonexistent inputs and an OCCUPIED destination: a stat would have refused
    with 4. Only the gate, read first, gives 5."""
    out = tmp_path / "estimate.json"
    out.write_text("prior")
    flags = [a for f in ("--pilot-manifest", "--study-manifest", "--artifact",
                         "--pilot-results", "--feasibility-report")
             for a in (f, str(tmp_path / "missing"))]
    segs = [a for k in range(4) for a in ("--segment", str(tmp_path / f"s{k}"))]
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h4_confirmatory_analysis",
                        *flags, *segs, "--out", str(out)], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == C.EXIT_UNAUTHORIZED, r.stderr
    assert out.read_text() == "prior" and os.listdir(tmp_path) == ["estimate.json"]


def test_the_gate_is_the_FIRST_statement_of_the_only_path_and_has_no_override():
    tree = ast.parse(pathlib.Path(C.__file__).read_text(encoding="utf-8"))
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run")
    first = run.body[0]
    assert isinstance(first, ast.If) and "H4_STUDY_AGGREGATION_AUTHORIZED" in ast.unparse(
        first.test)
    assert {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} \
        .isdisjoint({"environ", "getenv"})
    gates = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
             and n.id.endswith("_AUTHORIZED")}
    assert gates == {"H4_STUDY_AGGREGATION_AUTHORIZED"}


# ───────────────────── 2. the estimate, end to end (§3.2, §3.3) ─────────────────────

def test_an_ALL_CAP_study_counts_EVERY_identical_pair_and_cap_free_is_NOT_ESTIMABLE(
        study, gate_open):
    est = confirm(study)
    assert json.loads((study["repo"] / "estimate.json").read_text()) == est
    p = est["primary"]
    assert (p["n"], p["mean"]) == (296, 0.5), "296 identical pairs are 296 observations"
    assert est["concentration"]["pair_tuples"] == {"pairs": 296, "distinct": 1,
                                                   "max_frequency": 296}
    assert round(p["half_width"], 6) == 0.078938
    assert (p["lower"], p["upper"]) == (0.5 - p["half_width"], 0.5 + p["half_width"])
    assert p["reading"] == "contains" and "assumption, not a finding" in p["independence"]
    cf = est["cap_free_sensitivity"]
    assert cf == {"n": 0, "status": "not estimable", "label": C.CAP_FREE_LABEL}
    assert est["caps"] == {"cap_affected_pairs": 296, "cap_games": 592}
    assert est["fixture"] is True
    os.remove(study["repo"] / "estimate.json")                # module-shared repo


def test_a_MIXED_study_scores_every_pair_and_cap_free_drops_WHOLE_pairs(tmp_path,
                                                                       templates,
                                                                       gate_open):
    s = build_study(tmp_path / "repo", templates, picks=MIXED)
    est = confirm(s)
    picks = [MIXED[j % 4] for _seg in range(4) for j in range(74)]
    scores = [SCORE[p] for p in picks]
    assert est["primary"]["n"] == 296
    assert est["primary"]["mean"] == sum(scores) / 296
    kept = [SCORE[p] for p in picks if "cap" not in p]
    cf = est["cap_free_sensitivity"]
    assert (cf["n"], cf["mean"]) == (len(kept), sum(kept) / len(kept))
    assert cf["label"] == C.CAP_FREE_LABEL and "not a second confirmatory claim" in cf["label"]
    assert est["concentration"]["outcome_patterns"] == {
        "red|black": picks.count(("red", "black")), "cap|cap": picks.count(("cap", "cap")),
        "black|red": picks.count(("black", "red")), "red|cap": picks.count(("red", "cap"))}


def test_the_interval_is_RAW_beyond_1_when_every_pair_scores_1(tmp_path, templates,
                                                              gate_open):
    est = confirm(build_study(tmp_path / "repo", templates, picks=[("red", "black")]))
    p = est["primary"]
    assert p["mean"] == 1.0 and p["upper"] > 1.0, "never intersected with [0, 1]"
    assert p["reading"] == "above" and p["wording"] == C.READINGS["above"]


# ───────────────────────── 3. premature aggregation (§3.1) ─────────────────────────

def test_NO_ARTIFACT_is_refused(study, gate_open, tmp_path):
    with pytest.raises(F.H4AnalysisRefused, match="not durable"):
        confirm(study, artifact=str(tmp_path / "missing.json"), out=str(tmp_path / "e.json"))


@pytest.mark.parametrize("pilot_kw,decision", [
    ({"picks": [("red", "black")] * 16}, "STOP_COLLAPSE"),
    ({"total_s": 2391.0}, "STOP_RUNTIME"),
    ({"edit": lambda i, r: None if r["record_type"] == "segment_end" else r}, "VOID")])
def test_a_STOP_or_VOID_decision_is_refused(tmp_path, templates, gate_open, pilot_kw,
                                            decision):
    s = build_study(tmp_path / "repo", templates, picks=[("cap", "cap")], pilot_kw=pilot_kw)
    assert json.loads(s["artifact"].read_text())["decision"] == decision
    with pytest.raises(F.H4AnalysisRefused, match=f"says '{decision}', not PROCEED"):
        confirm(s)


@pytest.mark.parametrize("which", ["pilot_results", "feasibility_report"])
def test_an_artifact_whose_hash_no_longer_recomputes_is_refused(study, tmp_path, gate_open,
                                                                which):
    s = clone(study, tmp_path)
    s[which].write_text(s[which].read_text() + "\n")
    commit(s["repo"], s[which])                               # durable, but not what was decided
    with pytest.raises(F.H4AnalysisRefused, match="does not recompute"):
        confirm(s)


DURABLE = ["pilot_manifest", "study_manifest", "pilot_results", "feasibility_report",
           "artifact", "segment0", "segment1", "segment2", "segment3"]


@pytest.mark.parametrize("how", ["untracked", "modified"])
@pytest.mark.parametrize("which", DURABLE)
def test_EVERY_file_the_estimate_rests_on_must_be_COMMITTED_and_UNMODIFIED(
        study, tmp_path, gate_open, which, how):
    s = clone(study, tmp_path)
    target = (s["segments"][int(which[-1])] if which.startswith("segment") else s[which])
    if how == "untracked":
        git(s["repo"], "rm", "-q", "--cached", "--", str(target))
        commit(s["repo"])
    else:
        target.write_text(target.read_text() + "\n")
    with pytest.raises(F.H4AnalysisRefused, match=f"{target.name} is not durable"):
        confirm(s)
    assert not (s["repo"] / "estimate.json").exists()


def test_the_PUBLIC_entry_refuses_FIXTURE_input(study, gate_open, tmp_path):
    with pytest.raises(F.H4AnalysisRefused, match="negative seed"):
        C.run_confirmatory(**args(study, out=str(tmp_path / "e.json")))


def test_PILOT_data_in_a_segment_slot_is_refused(study, gate_open, tmp_path):
    segs = [str(study["pilot_results"])] + [str(p) for p in study["segments"][1:]]
    with pytest.raises(F.H4AnalysisRefused, match="differs from the manifest"):
        confirm(study, segments=segs, out=str(tmp_path / "e.json"))


@pytest.mark.parametrize("order,match", [
    ([0, 1, 2], "exactly 4"),
    ([0, 0, 2, 3], "segment 0 is not manifest slot 1"),
    ([1, 0, 2, 3], "segment 1 is not manifest slot 0")])
def test_MISSING_REPEATED_or_MISPLACED_segments_are_refused(study, gate_open, tmp_path,
                                                            order, match):
    segs = [str(study["segments"][k]) for k in order]
    with pytest.raises(F.H4AnalysisRefused, match=match):
        confirm(study, segments=segs, out=str(tmp_path / "e.json"))


def test_a_segment_from_ANOTHER_SCHEDULE_is_refused(study, tmp_path, templates, gate_open):
    s = clone(study, tmp_path)
    compose(s["segments"][2], templates, entry(2, 74, tag="elsewhere", seed0=90_000),
            ALL_CAP)
    commit(s["repo"], s["segments"][2])
    with pytest.raises(F.H4AnalysisRefused, match="schedule_digest"):
        confirm(s)


def test_a_STUDY_of_other_than_296_pairs_is_refused(study, tmp_path, gate_open):
    s = clone(study, tmp_path)
    entries = [entry(k, 74 if k else 73, tag="s", seed0=1000 * (k + 1)) for k in range(4)]
    s["study_manifest"].write_text(json.dumps({"stage": F.STUDY, "segments": entries}))
    commit(s["repo"], s["study_manifest"])
    with pytest.raises(F.H4AnalysisRefused, match="146 tasks, not 148"):
        confirm(s)


@pytest.mark.parametrize("edit,match", [
    (lambda i, r: {k: v for k, v in r.items() if k != "code"} if i == 0 else r,
     "no `code` field"),
    (lambda i, r: None if r["record_type"] == "segment_end" else r, "segment 2: "),
    (lambda i, r: dict(r, transcript_digest="0" * 64)
     if r["record_type"] == "game_result" and i > 300 else r, "segment 2: "),
    (lambda i, r: dict(r, seed=-77_777) if r["record_type"] == "game_start"
     and r["game_index"] == 9 else r, "not the manifest's schedule")])
def test_a_segment_that_is_not_complete_and_bound_is_REFUSED_not_voided(
        study, tmp_path, gate_open, edit, match):
    s = clone(study, tmp_path)
    edit_records(s["segments"][2], edit)
    commit(s["repo"], s["segments"][2])
    with pytest.raises(F.H4AnalysisRefused, match=match):
        confirm(s)
    assert not (s["repo"] / "estimate.json").exists()


# ──────────────────────────── 4. output semantics (§4.1) ────────────────────────────

def test_an_OCCUPIED_destination_is_refused_before_any_input(gate_open, tmp_path):
    out = tmp_path / "estimate.json"
    out.write_text("prior")
    with pytest.raises(F.H4AnalysisRefused, match="occupied"):
        C._run(pilot_manifest="m", study_manifest="s", artifact="a", pilot_results="r",
               feasibility_report="f", segments=[], out=str(out), _fixture=True)
    assert out.read_text() == "prior"


def test_a_refusal_leaves_no_output_and_no_temporary_file(study, tmp_path, gate_open):
    s = clone(study, tmp_path)
    before = sorted(os.listdir(s["repo"]))
    with pytest.raises(F.H4AnalysisRefused):
        confirm(s, segments=[str(p) for p in s["segments"][:3]])
    assert sorted(os.listdir(s["repo"])) == before


def test_NO_public_entry_can_reach_the_fixture_relaxation():
    assert fixture_leaks(C.__file__, {"run_confirmatory", "main"}) == []


# ───────────────────────────── 5. malformed pairs (§3.2) ─────────────────────────────

@pytest.fixture(scope="module")
def seg_games(study):
    return R.load_games(str(study["segments"][0]), allow_fixture=True)


def _copy(games):
    return json.loads(json.dumps(games))


def test_WELL_FORMED_pairs_are_all_formed_multiplicity_kept(seg_games):
    """CLEAN BASELINE: 74 identical pairs are 74 pairs."""
    assert len(C.form_pairs([_copy(seg_games)])) == 74


def _set(i, part, **kw):
    def edit(gs):
        gs[i][part].update(kw)
        return [gs]
    return edit


@pytest.mark.parametrize("mutate,match", [
    (lambda gs: [gs[:1] + gs[2:]], "1 games, not exactly two"),
    (_set(1, "start", arm="A"), "arms A, A"),
    (_set(0, "start", incumbent_colour="black", t1j_colour="red"), "are not Arm A's"),
    (_set(1, "start", game_index=3), "not adjacent"),
    (lambda gs: [gs[:1] + gs[2:], gs[1:2]], "crosses segments"),
    (_set(3, "start", seed=-1000), "seed -1000 appears 2 times"),
    (_set(3, "start", task_id="s0p0-A"), "task_id 's0p0-A' appears 2 times"),
    (lambda gs: [[dict(g, start=dict(g["start"], pair_id="s0p0"))
                  if g["start"]["pair_id"] == "s0p1" else g for g in gs]], "4 games"),
    (lambda gs: [[dict(gs[0], result=None)] + gs[1:]], "not complete"),
    (_set(0, "result", terminal_reason="budget"), "not complete"),
    (_set(0, "result", incumbent_points=1.0, t1j_points=0.0), "recorded points"),
    (lambda gs: [[dict(gs[0], plies=[dict(gs[0]["plies"][0], actor="t1j")]
                       + gs[0]["plies"][1:])] + gs[1:]], "actor disagrees")])
def test_a_MALFORMED_pair_is_an_integrity_fault_and_REFUSES_never_drops(seg_games, mutate,
                                                                       match):
    with pytest.raises(F.H4AnalysisRefused, match=match):
        C.form_pairs(mutate(_copy(seg_games)))


# ──────────────────────── 6. winner encoding, interval, cap-free (§3.2, §3.3) ────────────────────────

@pytest.mark.parametrize("result,colour,points", [
    ({"terminal_reason": "win", "winner": "red"}, "red", 1.0),
    ({"terminal_reason": "win", "winner": "black"}, "red", 0.0),
    ({"terminal_reason": "win", "winner": "black"}, "black", 1.0),
    ({"terminal_reason": "cap", "winner": None}, "black", 0.5)])
def test_the_FROZEN_winner_encoding(result, colour, points):
    assert C.incumbent_points(result, colour) == points


@pytest.mark.parametrize("result", [
    {"terminal_reason": "win", "winner": None}, {"terminal_reason": "cap", "winner": "red"},
    {"terminal_reason": "resign", "winner": "red"}, {"terminal_reason": "win",
                                                     "winner": "draw"}])
def test_a_winner_outside_the_encoding_is_an_integrity_fault(result):
    with pytest.raises(F.H4AnalysisRefused, match="integrity fault"):
        C.incumbent_points(result, "red")


def test_h_296_is_the_frozen_0_078938():
    assert round(C.half_width(296), 6) == 0.078938


@pytest.mark.parametrize("lower,upper,want", [
    (0.5, 0.9, "contains"), (0.5000001, 0.9, "above"), (0.1, 0.5, "contains"),
    (0.1, 0.4999999, "below"), (-0.05, 0.4, "below"), (0.6, 1.08, "above")])
def test_the_reading_is_STRICT_on_the_RAW_bounds(lower, upper, want):
    assert C.reading({"lower": lower, "upper": upper}) == want


def test_intervals_are_never_clipped_either_side():
    lo = C.interval([0.0] * 296)
    hi = C.interval([1.0] * 296)
    assert lo["lower"] < 0 and hi["upper"] > 1


def test_cap_free_with_NO_capless_pair_reports_no_mean_and_no_interval(seg_games):
    cf = C.estimate(C.form_pairs([_copy(seg_games)]))["cap_free_sensitivity"]
    assert cf["status"] == "not estimable" and cf["n"] == 0
    assert not {"mean", "lower", "upper", "half_width"} & set(cf)


# ─────────────────────── 7. H3 isolation and module separation (§0, §4) ───────────────────────

H3_NAMES = {"MAX_DUPLICATE_PAIRS", "MIN_DISTINCT_PER_CELL", "pair_identity"}
IMPORT_BANS = {"feasibility": {"h4_pilot_decision", "h4_confirmatory_analysis"},
               "decision": {"h4_confirmatory_analysis"}, "confirmatory": set()}


def isolation_faults(source, banned):
    tree = ast.parse(source)
    bad = []
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods = [n.module or ""] + [a.name for a in n.names]
        elif isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        for m in mods:
            leaf = m.split(".")[-1]
            if leaf.startswith("h3_") or leaf in banned:
                bad.append(f"import {m}")
        name = (n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute)
                else n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else None)
        if name and (name in H3_NAMES or "dup" in name.lower()):
            bad.append(f"name {name}")
    return bad


@pytest.mark.parametrize("key", list(MODULES))
def test_NO_H4_analysis_module_touches_H3_or_a_module_it_must_not(key):
    src = pathlib.Path(MODULES[key].__file__).read_text(encoding="utf-8")
    assert isolation_faults(src, IMPORT_BANS[key]) == []


@pytest.mark.parametrize("key,planted", [
    ("confirmatory", "from . import h3_study_analysis"),
    ("confirmatory", "import scripts.GPU.alphazero.h3_study_rules as H3"),
    ("confirmatory", "MAX_DUPLICATE_PAIRS = 3"),
    ("confirmatory", "def drop_duplicates(pairs):\n    return pairs"),
    ("confirmatory", "x = pair_identity"),
    ("decision", "from . import h4_confirmatory_analysis"),
    ("feasibility", "from . import h4_pilot_decision")])
def test_the_isolation_walker_is_NOT_VACUOUS(key, planted):
    """CLEAN-BASELINE CONTROL: each planted violation, in the REAL source, is seen."""
    src = pathlib.Path(MODULES[key].__file__).read_text(encoding="utf-8")
    assert isolation_faults(src + "\n" + planted + "\n", IMPORT_BANS[key]) != []
