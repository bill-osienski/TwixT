"""H5 confirmatory analysis (implementation card §6, §8; replication card §3-§6, §12).
SYNTHETIC FIXTURES ONLY.

Template games are PLAYED by the real H5 runner in fixture mode -- the H4 process-
boundary stubs, a stub incumbent, negative seeds -- and 4 x 74-pair segments are
COMPOSED from them inside a temporary git repository, beside a byte-exact copy of
the PINNED H4 estimate. No JVM, no research seed, no real outcome. The gate stays
CLOSED in the source; tests that need it open patch it for their own duration.
"""
import ast
import builtins
import inspect
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import h4_confirmatory_analysis as C4
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_repair_qualification as RQ
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_aggregation_authorization as AUTH
from scripts.GPU.alphazero import h5_confirmatory_analysis as C
from scripts.GPU.alphazero import h5_runner as H5
from scripts.GPU.alphazero import h5_study_authorization as RAUTH
from scripts.GPU.alphazero import t1j_adapter as A
from tests.test_h4_confirmatory_analysis import isolation_faults
from tests.test_h4_pilot_feasibility import (BLACK_WIN, CAP_AT, commit, edit_records,
                                             fixture_leaks, git, new_repo)
from tests.test_h4_repair_qualification import query_out
from tests.test_h4_runner import FAKE_TOOLCHAIN, SEQ, replay

ROOT = pathlib.Path(__file__).resolve().parents[1]
H4_EST = ROOT / C.H4_ESTIMATE
H4_PUBLISHED = json.loads(H4_EST.read_text())


def _play(seq, out_dir, cap=None):
    """One fixture pair through the REAL H5 runner: gate open, process boundary stubbed."""
    box = {"pid": 10_000_000}

    def fake_run(args, **kw):
        pairs = [tuple(int(v) for v in a.split(",")) for a in args
                 if "," in a and a.replace(",", "").isdigit()]
        prefix = [A.to_ours(x, y) for (x, y) in pairs]
        box["pid"] += 1
        if "replay" in args:
            return subprocess.CompletedProcess(args, 0, replay(prefix, pid=box["pid"]), "")
        native = len(prefix) <= 2
        return subprocess.CompletedProcess(
            args, RQ.NATIVE_EXIT if native else RQ.SEARCHED_EXIT,
            query_out(prefix, native=native, move=seq[len(prefix)], pid=box["pid"]), "")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", fake_run)
        mp.setattr(RAUTH, "H5_STUDY_EXECUTION_AUTHORIZED", True)
        if cap:
            mp.setattr(R4, "PLY_CAP", cap)
            mp.setattr(H5, "PLY_CAP", cap)
        H5.run_study(mode="fixture", schedule=H5.make_schedule([("t", -1, -2)], mode="fixture"),
                     out_dir=out_dir, deadline_s=600,
                     paths=H5.T1jPaths(java="/j", jar="/x.jar", classes=out_dir + ".cls",
                                       ply_cap=280),
                     incumbent_build=lambda t, evaluator=None: (lambda s: seq[s.ply]),
                     incumbent_identity={"stub": True},
                     _compile=lambda d: dict(FAKE_TOOLCHAIN))
    return os.path.join(out_dir, "results.jsonl")


@pytest.fixture(scope="session")
def templates(tmp_path_factory):
    """red: Red wins · black: Black wins · cap: capped. Arm A = incumbent Red."""
    d = tmp_path_factory.mktemp("h5_templates")
    out = {}
    for name, seq, cap in (("red", SEQ, None), ("black", BLACK_WIN, None),
                           ("cap", SEQ, CAP_AT)):
        recs = R4.read_records(_play(seq, str(d / name), cap))
        assert recs[0]["design"] == "H5_SYNTHETIC_FIXTURE" and recs[-1]["record_type"] == \
            "segment_end"
        out[name] = {"header": recs[0],
                     "A": [r for r in recs if r.get("task_id") == "t-A"],
                     "B": [r for r in recs if r.get("task_id") == "t-B"]}
    return out


def entry(segment, *, tag="s", seed0=1, positive=False):
    sign = 1 if positive else -1
    pairs = [(f"{tag}{segment}p{k}", sign * (seed0 + 2 * k), sign * (seed0 + 2 * k + 1))
             for k in range(H5.PAIRS_PER_SEGMENT)]
    sched = R4.make_schedule(pairs, mode="study" if positive else "fixture")
    return {"segment": segment, "schedule": sched, "schedule_digest": R4.schedule_digest(sched),
            "seeds": [t["seed"] for t in sched], "incumbent_identity": {"stub": True},
            "t1j_runtime": {"toolchain": {"stub": True}, "depth": R4.DEPTH,
                            "query_timeout_s": R4.QUERY_TIMEOUT_S,
                            "replay_timeout_s": R4.REPLAY_TIMEOUT_S, "ply_cap": R4.PLY_CAP,
                            "h4_acceptance": True},
            "cards": {c: "c" * 64 for c in H5.CARDS},
            "code": {m: "d" * 64 for m in H5.CODE}}


HEADER_FIELDS = ("schedule_digest", "seeds", "incumbent_identity", "t1j_runtime", "cards",
                 "code")


def compose(path, templates, e, picks, *, design="H5_SYNTHETIC_FIXTURE", total_s=2390.0):
    """A results file for entry `e`; `picks[k]` = (Arm A template, Arm B template)."""
    lines = [dict(templates["red"]["header"], design=design,
                  evidence=design == "H5_STUDY", segment=e["segment"],
                  **{k: e[k] for k in HEADER_FIELDS})]
    for i, task in enumerate(e["schedule"]):
        name = picks[i // 2][0 if task["arm"] == "A" else 1]
        for r in templates[name][task["arm"]]:
            r = dict(r, task_id=task["task_id"], pair_id=task["pair_id"])
            if r["record_type"] == "game_start":
                r.update(seed=task["seed"], game_index=i)
            lines.append(r)
    lines.append({"record_type": "segment_end", "games_completed": len(e["schedule"]),
                  "complete": True, "setup_s": 1.0, "total_s": total_s})
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in lines))


ALL_CAP = [("cap", "cap")]
#: pair scores 1, 0.5, 0: incumbent points (A) + (B), halved
SCORE = {("red", "black"): 1.0, ("cap", "cap"): 0.5, ("black", "red"): 0.0}


def build_study(root, templates, picks=ALL_CAP, *, per_segment=None, positive=False,
                design="H5_SYNTHETIC_FIXTURE"):
    repo = new_repo(root)
    est = repo / C.H4_ESTIMATE
    est.parent.mkdir(parents=True)
    shutil.copy2(H4_EST, est)                           # byte-exact: the pin recomputes
    sd = repo / "study"
    sd.mkdir()
    entries = [entry(k, seed0=1000 * (k + 1), positive=positive) for k in range(4)]
    man = sd / "manifest.json"
    man.write_text(json.dumps({"stage": "H5_STUDY", "segments": entries}))
    segs = []
    for k, e in enumerate(entries):
        p = (per_segment or {}).get(k, picks)
        segs.append(sd / f"segment{k}.jsonl")
        compose(segs[-1], templates, e, [p[j % len(p)] for j in range(74)], design=design,
                total_s=2000.0 + k)
    commit(repo, est, man, *segs)
    return {"repo": repo, "study_manifest": man, "segments": segs, "h4_estimate": est,
            "entries": entries}


def args(study, **over):
    kw = {"study_manifest": str(study["study_manifest"]),
          "segments": [str(s) for s in study["segments"]],
          "h4_estimate": str(study["h4_estimate"]), "out": str(study["repo"] / "e.json")}
    kw.update(over)
    return kw


def confirm(study, **over):
    return C._run(**args(study, **over), _fixture=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory, templates):
    return build_study(tmp_path_factory.mktemp("h5study") / "repo", templates)


def clone(study, tmp_path):
    dst = tmp_path / "repo"
    shutil.copytree(study["repo"], dst)
    move = lambda p: dst / pathlib.Path(p).relative_to(study["repo"])   # noqa: E731
    return {"repo": dst, "study_manifest": move(study["study_manifest"]),
            "segments": [move(s) for s in study["segments"]],
            "h4_estimate": move(study["h4_estimate"]), "entries": study["entries"]}


@pytest.fixture
def gate_open(monkeypatch):
    monkeypatch.setattr(AUTH, "H5_STUDY_AGGREGATION_AUTHORIZED", True)


@pytest.fixture
def fresh_destination(monkeypatch, tmp_path):
    """TEST-LOCAL. Since step 13 the committed estimate occupies the real `C.OUT`, so
    an in-process test that must reach a check AFTER the destination's gets a fresh
    temporary one. Absolute, so the public entry's join with REPO_ROOT yields it."""
    out = tmp_path / "fresh-destination" / "estimate.json"
    monkeypatch.setattr(C, "OUT", str(out))
    return out


# ─────────────────────────── 1. THE GATE comes first (§6.1) ───────────────────────────

class Spy:
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


def test_the_gate_is_false_as_published():
    assert AUTH.H5_STUDY_AGGREGATION_AUTHORIZED is False


def test_a_CLOSED_gate_opens_stats_hashes_and_runs_NOTHING(study, monkeypatch,
                                                         fresh_destination):
    spy = Spy(monkeypatch, study["repo"])
    with pytest.raises(C.H5AggregationUnauthorized):
        C.run_confirmatory(study_manifest=str(study["study_manifest"]),
                           segments=[str(s) for s in study["segments"]])
    with pytest.raises(C.H5AggregationUnauthorized):
        C._run(**args(study), _fixture=True)
    assert spy.seen == []
    assert not (study["repo"] / "e.json").exists()


def test_the_input_spy_is_NOT_VACUOUS(study, monkeypatch, gate_open):
    spy = Spy(monkeypatch, study["repo"])
    with pytest.raises(F.H4AnalysisRefused):
        C._run(**args(study, segments=[str(s) for s in study["segments"][:3]]),
               _fixture=True)
    assert {"lstat"} <= {n for n, _p in spy.seen}
    spy.seen.clear()
    with pytest.raises(F.H4AnalysisRefused):
        C._run(**args(study, h4_estimate=str(study["study_manifest"])), _fixture=True)
    assert {"lstat", "run", "open"} <= {n for n, _p in spy.seen}


def test_the_CLI_with_the_gate_closed_exits_5_before_touching_ANY_path(tmp_path):
    """A fresh subprocess with the PRODUCTION gate closed. Since step 13 the real
    destination holds the committed estimate: it must be left exactly as it was."""
    real = ROOT / C.OUT

    def state():
        return ((real.read_bytes(), real.stat().st_mtime_ns) if real.exists() else None,
                sorted(os.listdir(real.parent)) if real.parent.exists() else None)
    before = state()
    segs = [a for k in range(4) for a in ("--segment", str(tmp_path / f"s{k}"))]
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h5_confirmatory_analysis",
                        "--study-manifest", str(tmp_path / "missing"), *segs],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == C.EXIT_UNAUTHORIZED == 5, r.stderr
    assert os.listdir(tmp_path) == []
    assert state() == before, "the CLI changed the real destination"


def test_the_gate_is_the_FIRST_statement_of_the_only_path_and_has_no_override():
    tree = ast.parse(pathlib.Path(C.__file__).read_text(encoding="utf-8"))
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run")
    first = run.body[0]
    assert isinstance(first, ast.If) and \
        ast.unparse(first.test) == "not AUTH.H5_STUDY_AGGREGATION_AUTHORIZED"
    assert {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} \
        .isdisjoint({"environ", "getenv"})
    assert not {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                and n.id.endswith("_AUTHORIZED")}


# ───────────────────── 2. the estimate and the verdict (§6.2) ─────────────────────

def test_an_ALL_CAP_study_is_INCONCLUSIVE_with_every_identical_pair_counted(study,
                                                                           gate_open):
    est = confirm(study)
    assert json.loads((study["repo"] / "e.json").read_text()) == est
    p = est["primary"]
    assert (p["n"], p["mean"]) == (296, 0.5), "296 identical pairs are 296 observations"
    assert round(p["half_width"], 10) == 0.0789380472
    assert p["verdict"] == "INCONCLUSIVE" and p["wording"] == C.WORDING["INCONCLUSIVE"]
    assert "assumption, not an observed property" in p["independence"]
    assert "fresh, non-overlapping" in p["independence"]
    assert p["h4"] == "H4's published result stands unchanged whatever H5 finds"
    assert est["concentration"]["pair_tuples"] == {"pairs": 296, "distinct": 1,
                                                   "max_frequency": 296}
    assert est["cap_free_sensitivity"] == {"n": 0, "status": "not estimable",
                                           "label": C.CAP_FREE_LABEL}
    assert est["caps"] == {"cap_affected_pairs": 296, "cap_games": 592}
    os.remove(study["repo"] / "e.json")                       # module-shared repo


@pytest.mark.parametrize("picks,verdict,mean", [
    ([("red", "black")], "DIRECTIONAL_REPLICATION", 1.0),
    ([("black", "red")], "OPPOSITE_DIRECTION", 0.0)])
def test_a_one_sided_study_gets_its_verdict_on_RAW_bounds(tmp_path, templates, gate_open,
                                                          picks, verdict, mean):
    est = confirm(build_study(tmp_path / "repo", templates, picks=picks))
    p = est["primary"]
    assert (p["mean"], p["verdict"]) == (mean, verdict)
    assert p["upper"] > 1.0 if mean == 1.0 else p["lower"] < 0.0, "never clipped"


@pytest.mark.parametrize("lower,upper,want", [
    (0.5, 0.9, "INCONCLUSIVE"), (0.5000001, 0.9, "DIRECTIONAL_REPLICATION"),
    (0.1, 0.5, "INCONCLUSIVE"), (0.1, 0.4999999, "OPPOSITE_DIRECTION"),
    (0.49, 0.51, "INCONCLUSIVE")])
def test_the_verdict_is_STRICT_on_the_RAW_bounds(lower, upper, want):
    assert C.verdict({"lower": lower, "upper": upper}) == want


def test_the_verdict_uses_the_COMPUTED_bound_not_a_rounded_threshold():
    """686/1184 is the smallest attainable replicating mean, 685/1184 the largest
    that is not (replication card §3.1); the rounded 0.578938 sits between them."""
    h = C4.half_width(296)
    for j, want in ((686, "DIRECTIONAL_REPLICATION"), (685, "INCONCLUSIVE")):
        m = j / 1184
        assert C.verdict({"lower": m - h, "upper": m + h}) == want


def test_PER_SEGMENT_results_are_reported_IN_RUN_ORDER_and_DESCRIPTIVE_only(tmp_path,
                                                                            templates,
                                                                            gate_open):
    s = build_study(tmp_path / "repo", templates, per_segment={
        0: [("red", "black")], 1: [("cap", "cap")], 2: [("black", "red")],
        3: [("red", "black"), ("black", "red")]})
    est = confirm(s)
    ps = est["per_segment"]
    assert ps["label"] == C.PER_SEGMENT_LABEL
    assert [x["segment"] for x in ps["segments"]] == [0, 1, 2, 3]
    assert [x["mean"] for x in ps["segments"]] == [1.0, 0.5, 0.0, 0.5]
    assert [x["cap_affected_pairs"] for x in ps["segments"]] == [0, 74, 0, 0]
    assert [x["total_s"] for x in ps["segments"]] == [2000.0, 2001.0, 2002.0, 2003.0]
    assert all(set(x) == {"segment", "pairs", "mean", "cap_affected_pairs", "cap_games",
                          "total_s"} for x in ps["segments"]), "no per-segment interval"
    assert est["primary"]["mean"] == 0.5
    cf = est["cap_free_sensitivity"]
    assert (cf["n"], cf["mean"]) == (222, 0.5) and "not a second confirmatory" in cf["label"]


def test_H4_BESIDE_H5_is_read_from_the_PINNED_estimate_and_is_DESCRIPTIVE(study,
                                                                        gate_open):
    est = confirm(study, out=str(study["repo"] / "e2.json"))
    side = est["h4_beside_h5"]
    assert side["label"] == "DESCRIPTIVE -- no interval, no test, no pooled estimate"
    h4 = H4_PUBLISHED["primary"]
    assert {k: side["h4"][k] for k in ("n", "mean", "lower", "upper")} == \
        {k: h4[k] for k in ("n", "mean", "lower", "upper")}
    assert side["h4"]["caps"] == H4_PUBLISHED["caps"]
    assert side["h4"]["estimate_sha256"] == C.H4_ESTIMATE_SHA256
    assert side["difference_of_means"] == 0.5 - h4["mean"]
    assert set(side) == {"label", "h4", "h5", "difference_of_means"}
    assert est["inputs"][str(study["h4_estimate"])] == C.H4_ESTIMATE_SHA256
    os.remove(study["repo"] / "e2.json")


def test_the_OUTPUT_carries_exactly_the_frozen_blocks_and_nothing_pooled(study, gate_open):
    est = confirm(study, out=str(study["repo"] / "e3.json"))
    assert set(est) == {"analysis", "fixture", "primary", "cap_free_sensitivity", "caps",
                        "per_segment", "concentration", "h4_beside_h5", "inputs",
                        "written_at"}
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k).lower()
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    every = set(keys({k: v for k, v in est.items() if k != "inputs"}))   # inputs: paths
    for banned in ("pool", "p_value", "combined", "z_score", "test"):
        assert not [k for k in every if banned in k], banned
    os.remove(study["repo"] / "e3.json")


# ───────────────────────── 3. what refuses (§6.1) ─────────────────────────

DURABLE = ["study_manifest", "h4_estimate", "segment0", "segment1", "segment2", "segment3"]


@pytest.mark.parametrize("how", ["untracked", "modified"])
@pytest.mark.parametrize("which", DURABLE)
def test_EVERY_input_must_be_COMMITTED_and_UNMODIFIED(study, tmp_path, gate_open, which, how):
    s = clone(study, tmp_path)
    target = s["segments"][int(which[-1])] if which.startswith("segment") else s[which]
    if how == "untracked":
        git(s["repo"], "rm", "-q", "--cached", "--", str(target))
        commit(s["repo"])
    else:
        target.write_text(target.read_text() + "\n")
    with pytest.raises(F.H4AnalysisRefused, match=f"{target.name} is not durable"):
        confirm(s)
    assert not (s["repo"] / "e.json").exists()


def test_an_H4_ESTIMATE_that_does_not_hash_to_its_PIN_is_refused(study, tmp_path, gate_open):
    s = clone(study, tmp_path)
    s["h4_estimate"].write_text(s["h4_estimate"].read_text().replace("0.5929", "0.6929", 1))
    commit(s["repo"], s["h4_estimate"])                     # durable, but not the one pinned
    with pytest.raises(F.H4AnalysisRefused, match="does not hash to the pinned H4 estimate"):
        confirm(s)


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
    compose(s["segments"][2], templates, entry(2, tag="elsewhere", seed0=90_000), ALL_CAP * 74)
    commit(s["repo"], s["segments"][2])
    with pytest.raises(F.H4AnalysisRefused, match="schedule_digest"):
        confirm(s)


def test_a_game_that_is_NOT_the_manifests_schedule_is_refused(study, tmp_path, gate_open):
    s = clone(study, tmp_path)
    edit_records(s["segments"][1], lambda i, r: dict(r, seed=-77_777)
                 if r["record_type"] == "game_start" and r["game_index"] == 9 else r)
    commit(s["repo"], s["segments"][1])
    with pytest.raises(F.H4AnalysisRefused, match="not the manifest's schedule"):
        confirm(s)


@pytest.mark.parametrize("stage", ["H4_STUDY", "H4_PILOT", "H5_SYNTHETIC_FIXTURE"])
def test_a_manifest_of_ANOTHER_STAGE_is_refused(study, tmp_path, gate_open, stage):
    s = clone(study, tmp_path)
    m = json.loads(s["study_manifest"].read_text())
    s["study_manifest"].write_text(json.dumps(dict(m, stage=stage)))
    commit(s["repo"], s["study_manifest"])
    with pytest.raises(F.H4AnalysisRefused, match="an H5_STUDY manifest is required"):
        confirm(s)


@pytest.mark.parametrize("field", ["cards", "code"])
def test_a_manifest_binding_H4s_cards_or_code_LIST_is_refused(study, tmp_path, gate_open,
                                                             field):
    s = clone(study, tmp_path)
    m = json.loads(s["study_manifest"].read_text())
    src = R4.CARDS if field == "cards" else R4.CODE
    for e in m["segments"]:
        e[field] = {p: "0" * 64 for p in src}
    s["study_manifest"].write_text(json.dumps(m))
    commit(s["repo"], s["study_manifest"])
    with pytest.raises(F.H4AnalysisRefused, match=f"{field} must be exactly H5's"):
        confirm(s)


def test_H4s_REAL_study_manifest_is_refused_as_H5_input(study, gate_open, tmp_path):
    with pytest.raises(F.H4AnalysisRefused, match="required, not 'H4_STUDY'"):
        confirm(study, study_manifest=str(ROOT / "docs/superpowers/evidence/"
                                          "t1j-h4-manifests/study_manifest.json"),
                out=str(tmp_path / "e.json"))


def test_an_H4_RESULTS_FILE_in_a_segment_slot_is_refused(study, gate_open, tmp_path):
    h4seg = ROOT / "docs/superpowers/evidence/t1j-h4-study-segment0/results.jsonl"
    segs = [str(h4seg)] + [str(p) for p in study["segments"][1:]]
    with pytest.raises(F.H4AnalysisRefused, match="design 'H4_STUDY'"):
        confirm(study, segments=segs, out=str(tmp_path / "e.json"))


def test_H4s_CONFIRMATORY_refuses_an_H5_manifest(study, monkeypatch, tmp_path):
    """The other direction: H4's analysis, gate open, handed H5's manifest after its
    own real pilot chain, refuses on the stage."""
    monkeypatch.setattr(C4, "H4_STUDY_AGGREGATION_AUTHORIZED", True)
    ev = ROOT / "docs/superpowers/evidence"
    with pytest.raises(F.H4AnalysisRefused, match="a H4_STUDY manifest is required, not "
                                                  "'H5_STUDY'"):
        C4._run(pilot_manifest=str(ev / "t1j-h4-manifests/pilot_manifest.json"),
                study_manifest=str(study["study_manifest"]),
                artifact=str(ev / "t1j-h4-pilot/decision.json"),
                pilot_results=str(ev / "t1j-h4-pilot/results.jsonl"),
                feasibility_report=str(ev / "t1j-h4-pilot/feasibility.json"),
                segments=[str(s) for s in study["segments"]],
                out=str(tmp_path / "e.json"), _fixture=True)


def test_NO_PILOT_ARTIFACT_is_accepted_or_even_nameable():
    for fn in (C.run_confirmatory, C._run):
        params = set(inspect.signature(fn).parameters)
        assert not {p for p in params if any(w in p for w in ("pilot", "artifact",
                                                               "feasibility"))}, params
    assert set(inspect.signature(C.run_confirmatory).parameters) == {"study_manifest",
                                                                     "segments"}


# ─────────── 4. 🔴 the H5 FIXTURE label is refused by H5's OWN binding (§6.1, review) ───────────

def test_bind_header_REFUSES_an_H5_FIXTURE_header_on_the_public_path(templates):
    e = entry(0)
    header = dict(templates["red"]["header"], segment=0, **{k: e[k] for k in HEADER_FIELDS})
    assert header["design"] == "H5_SYNTHETIC_FIXTURE"
    with pytest.raises(F.H4AnalysisRefused, match="design 'H5_SYNTHETIC_FIXTURE', not "
                                                  "'H5_STUDY'"):
        C.bind_header(header, e, _fixture=False)
    C.bind_header(header, e, _fixture=True)                 # CLEAN BASELINE: fixture path


def test_the_PUBLIC_ENTRY_refuses_an_H5_FIXTURE_file_at_its_header_binding(tmp_path,
                                                                          templates,
                                                                          gate_open,
                                                                          monkeypatch):
    """DIRECT NEGATIVE CONTROL. Everything else about this study passes the public
    path -- durable inside the (patched) repository, the pinned H4 estimate, an
    H5_STUDY manifest with positive seeds -- so the ONLY thing refusing it is the
    `H5_SYNTHETIC_FIXTURE` label. `load_games` alone would have accepted it."""
    s = build_study(tmp_path / "repo", templates, positive=True)
    monkeypatch.setattr(R4, "REPO_ROOT", s["repo"].resolve())
    assert len(R4.load_games(str(s["segments"][0]))) == 148, "H4's reader lets it through"
    with pytest.raises(F.H4AnalysisRefused, match="design 'H5_SYNTHETIC_FIXTURE', not "
                                                  "'H5_STUDY'"):
        C.run_confirmatory(study_manifest=str(s["study_manifest"]),
                           segments=[str(p) for p in s["segments"]])
    assert not (s["repo"] / C.OUT).parent.exists(), "a refusal writes nothing"


def test_the_same_study_labelled_H5_STUDY_PASSES_the_public_path(tmp_path, templates,
                                                                gate_open, monkeypatch):
    """CLEAN BASELINE for the control above: the label is the only difference."""
    s = build_study(tmp_path / "repo", templates, positive=True, design="H5_STUDY")
    monkeypatch.setattr(R4, "REPO_ROOT", s["repo"].resolve())
    est = C.run_confirmatory(study_manifest=str(s["study_manifest"]),
                             segments=[str(p) for p in s["segments"]])
    assert est["fixture"] is False and est["primary"]["n"] == 296
    assert (s["repo"] / C.OUT).exists()


def test_the_PUBLIC_entry_refuses_FIXTURE_CONTENT_even_inside_the_repository(
        study, gate_open, monkeypatch):
    monkeypatch.setattr(R4, "REPO_ROOT", study["repo"].resolve())
    with pytest.raises(F.H4AnalysisRefused, match="negative seed"):
        C.run_confirmatory(study_manifest=str(study["study_manifest"]),
                           segments=[str(p) for p in study["segments"]])


def test_the_PUBLIC_entry_refuses_input_outside_THIS_repository(study, gate_open,
                                                                fresh_destination):
    with pytest.raises(F.H4AnalysisRefused, match="does not resolve into this repository"):
        C.run_confirmatory(study_manifest=str(study["study_manifest"]),
                           segments=[str(p) for p in study["segments"]])
    assert not fresh_destination.parent.exists(), "a refusal writes nothing"


def test_NO_public_entry_can_reach_the_fixture_relaxation():
    assert fixture_leaks(C.__file__, {"run_confirmatory", "main"}) == []


# ──────────────────────────── 5. output semantics ────────────────────────────

def test_an_OCCUPIED_destination_is_refused_before_any_input(gate_open, tmp_path):
    out = tmp_path / "estimate.json"
    out.write_text("prior")
    with pytest.raises(F.H4AnalysisRefused, match="occupied"):
        C._run(study_manifest="m", segments=[], h4_estimate="h", out=str(out),
               _fixture=True)
    assert out.read_text() == "prior"


def test_a_refusal_leaves_no_output_no_directory_and_no_temporary_file(study, tmp_path,
                                                                      gate_open):
    s = clone(study, tmp_path)
    before = sorted(os.listdir(s["repo"]))
    with pytest.raises(F.H4AnalysisRefused):
        confirm(s, segments=[str(p) for p in s["segments"][:3]],
                out=str(s["repo"] / "new-dir" / "e.json"))
    assert sorted(os.listdir(s["repo"])) == before


# ─────────────── 6. isolation: no H3, no H4 wording, no pooling (§6.2) ───────────────

def test_the_H5_analysis_touches_no_H3_and_no_H4_VERDICT_code():
    src = pathlib.Path(C.__file__).read_text(encoding="utf-8")
    assert isolation_faults(src, set()) == []
    tree = ast.parse(src)
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
            and getattr(n.value, "id", None) == "C4"}
    assert not used & {"estimate", "READINGS", "reading"}, used
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not {x for x in names if "pool" in x.lower() or "combine" in x.lower()}


@pytest.mark.parametrize("planted", ["from . import h3_study_analysis",
                                     "x = C4.estimate", "def pooled_mean(a, b):\n    pass"])
def test_the_isolation_checks_are_NOT_VACUOUS(planted):
    src = pathlib.Path(C.__file__).read_text(encoding="utf-8") + "\n" + planted + "\n"
    tree = ast.parse(src)
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
            and getattr(n.value, "id", None) == "C4"}
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert isolation_faults(src, set()) or used & {"estimate"} or \
        {x for x in names if "pool" in x.lower()}
