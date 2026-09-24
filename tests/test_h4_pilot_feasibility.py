"""H4 pilot feasibility (analysis card §1, §2.1, §4, §4.1). SYNTHETIC FIXTURES ONLY.

No JVM, no research seed, no research game, no real outcome. Four template games
are PLAYED by the real H4 runner in fixture mode -- the same process-boundary
stubs as `tests/test_h4_runner.py` -- and full-size pilot and study files are
COMPOSED from them: negative seeds, `H4_SYNTHETIC_FIXTURE` headers, temporary git
repositories. The composed headers carry the `code` field the step-2 runner does
not yet write (card §0.1); a header without it is refused.

The fixture relaxation (`_fixture=True`) is reached only through the private
cores; a control below proves neither public entry can reach it.
"""
import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import h4_pilot_authorization as AUTH
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_repair_qualification as Q
from scripts.GPU.alphazero import h4_runner as R
from scripts.GPU.alphazero import t1j_adapter as A
from tests.test_h4_repair_qualification import query_out
from tests.test_h4_runner import FAKE_TOOLCHAIN, SEQ, replay

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A 26-ply game BLACK wins (a knight's-move chain from column 0 to column 23),
#: Red filling row 20/22 where it can neither win nor cut a black bridge.
BLACK_WIN = [m for pair in zip(
    [(20, 1), (20, 3), (20, 5), (20, 7), (20, 9), (20, 11), (20, 13), (20, 15), (20, 17),
     (20, 19), (20, 21), (22, 1), (22, 3)],
    [(11, 0), (12, 2), (11, 4), (12, 6), (11, 8), (12, 10), (11, 12), (12, 14), (11, 16),
     (12, 18), (11, 20), (12, 22), (10, 23)]) for m in pair]
#: SEQ with one Black move changed: a DIFFERENT Red-win trajectory.
SEQ2 = SEQ[:-2] + [(7, 22), SEQ[-1]]
CAP_AT = 6


def _play(seq, out_dir, cap=None):
    """One fixture pair through the REAL runner: gate open, process boundary stubbed."""
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
            args, Q.NATIVE_EXIT if native else Q.SEARCHED_EXIT,
            query_out(prefix, native=native, move=seq[len(prefix)], pid=box["pid"]), "")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", fake_run)
        mp.setattr(AUTH, "H4_PILOT_EXECUTION_AUTHORIZED", True)
        if cap:
            mp.setattr(R, "PLY_CAP", cap)
        R.run_games(mode="fixture", schedule=R.make_schedule([("t", -1, -2)], mode="fixture"),
                    out_dir=out_dir, deadline_s=600,
                    paths=R.T1jPaths(java="/j", jar="/x.jar", classes=out_dir + ".cls",
                                     ply_cap=280),
                    incumbent_build=lambda t, evaluator=None: (lambda s: seq[s.ply]),
                    incumbent_identity={"stub": True},
                    _compile=lambda d: dict(FAKE_TOOLCHAIN))
    return os.path.join(out_dir, "results.jsonl")


@pytest.fixture(scope="session")
def templates(tmp_path_factory):
    """red / red2: Red wins · black: Black wins · cap: capped at CAP_AT plies.
    Arm A is the incumbent in Red, Arm B the incumbent in Black."""
    d = tmp_path_factory.mktemp("h4_templates")
    out = {}
    for name, seq, cap in (("red", SEQ, None), ("red2", SEQ2, None),
                           ("black", BLACK_WIN, None), ("cap", SEQ, CAP_AT)):
        recs = R.read_records(_play(seq, str(d / name), cap))
        out[name] = {"header": recs[0],
                     "A": [r for r in recs if r.get("task_id") == "t-A"],
                     "B": [r for r in recs if r.get("task_id") == "t-B"]}
    return out


def entry(segment, n_pairs, *, tag, seed0=1):
    sched = R.make_schedule([(f"{tag}{segment}p{k}", -(seed0 + 2 * k), -(seed0 + 2 * k + 1))
                             for k in range(n_pairs)], mode="fixture")
    return {"segment": segment, "schedule": sched, "schedule_digest": R.schedule_digest(sched),
            "seeds": [t["seed"] for t in sched], "incumbent_identity": {"stub": True},
            "t1j_runtime": {"toolchain": {"stub": True}, "depth": R.DEPTH,
                            "query_timeout_s": R.QUERY_TIMEOUT_S,
                            "replay_timeout_s": R.REPLAY_TIMEOUT_S, "ply_cap": R.PLY_CAP,
                            "h4_acceptance": True},
            "cards": {c: "c" * 64 for c in F.MANIFEST_CARDS},
            "code": {m: "d" * 64 for m in F.MANIFEST_CODE}}


#: Card §0.1's bound fields, spelled out HERE rather than read from `F.BOUND`: a
#: fixture built from the module's own list would lose a field the module lost.
HEADER_FIELDS = ("schedule_digest", "seeds", "incumbent_identity", "t1j_runtime", "cards",
                 "code")


def compose(path, templates, e, picks, *, setup_s=70.0, total_s=2390.0, end=True):
    """A results file for manifest entry `e`; `picks[k]` = (Arm A template, Arm B template)."""
    lines = [dict(templates["red"]["header"], design=R.FIXTURE_DESIGN, evidence=False,
                  segment=e["segment"], **{k: e[k] for k in HEADER_FIELDS})]
    for i, task in enumerate(e["schedule"]):
        name = picks[i // 2][0 if task["arm"] == "A" else 1]
        for r in templates[name][task["arm"]]:
            r = dict(r, task_id=task["task_id"], pair_id=task["pair_id"])
            if r["record_type"] == "game_start":
                r.update(seed=task["seed"], game_index=i)
            lines.append(r)
    if end:
        lines.append({"record_type": "segment_end", "games_completed": len(e["schedule"]),
                      "complete": True, "setup_s": setup_s, "total_s": total_s})
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in lines))


def edit_records(path, fn):
    recs = R.read_records(str(path))
    out = [r for r in (fn(i, r) for i, r in enumerate(recs)) if r is not None]
    pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in out))


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True)


def new_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    return path


def commit(repo, *paths):
    if paths:
        git(repo, "add", "--", *map(str, paths))
    git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-q", "-m", "synthetic fixture")


PROCEED_PICKS = [("red2", "black")] + [("red", "black")] * 15


def make_pilot(repo, templates, picks=PROCEED_PICKS, *, name="pilot", setup_s=70.0,
               total_s=2390.0, edit=None, manifest_entry=None, manifest=None,
               to_commit=("manifest", "results")):
    d = repo / name
    d.mkdir()
    e = entry(0, 16, tag=name)
    man, res = d / "manifest.json", d / "results.jsonl"
    man.write_text(json.dumps(manifest if manifest is not None else
                              {"stage": F.PILOT, "segments": [manifest_entry or e]}))
    compose(res, templates, e, picks, setup_s=setup_s, total_s=total_s)
    if edit:
        edit_records(res, edit)
    files = {"manifest": man, "results": res}
    if to_commit:
        commit(repo, *(files[k] for k in to_commit))
    return man, res


def feasibility(man, res, out=None):
    out = out or pathlib.Path(res).parent / "feasibility.json"
    return F._write_report(str(man), str(res), str(out), _fixture=True)


@pytest.fixture
def repo(tmp_path):
    return new_repo(tmp_path / "repo")


# ─────────────────────────── 1. the report, and the rules (§1) ───────────────────────────

def test_a_clean_pilot_is_PROCEED_eligible_and_the_report_is_what_was_written(repo, templates):
    man, res = make_pilot(repo, templates)
    rep = feasibility(man, res)
    assert (rep["verdict"], rep["rules_fired"]) == ("PROCEED", [])
    assert json.loads((repo / "pilot" / "feasibility.json").read_text()) == rep
    assert rep["operational"]["process_liveness"] == (
        "checked by the runner at the end of each game (a live recorded pid VOIDs the "
        "run before segment_end is written); it cannot be measured retrospectively "
        "from this report")
    assert sorted(c for c in rep["concentration"]["by_arm"]["A"]["prefix_frequencies"]) \
        == ["1", "2", "3", "4", "5", "6"], "k = 1..6, frozen by the card"
    assert rep["runtime"]["mean_game_s"] == 72.5
    assert rep["runtime"]["projected_segment_s"] == 10_800
    assert rep["caps"] == {"cap_affected_pairs": 0, "cap_games": 0, "stop_at": 4}
    assert len(rep["games"]) == 32 and all(g["all_accepted"] for g in rep["games"])
    c = rep["concentration"]
    assert c["pair_tuples"] == {"pairs": 16, "distinct": 2, "max_frequency": 15}
    assert c["by_arm"]["A"]["unique_games"] == 2 and c["by_arm"]["B"]["unique_games"] == 1
    assert c["by_arm"]["A"]["prefix_frequencies"]["1"] == {"0,4": 16}
    # T1j is Black in Arm A (even plies of 25) and Red in Arm B (odd plies of 26);
    # the stub answers plies 1-3 natively, as the qualified helper does.
    src = {1: Q.NATIVE_FIRST, 2: Q.NATIVE_SECOND_TO_FOURTH, 3: Q.NATIVE_SECOND_TO_FOURTH}
    assert c["routines_by_ply_and_colour"] == {
        str(p): {"red" if p % 2 else "black": {src.get(p, Q.SEARCHED): 16}}
        for p in range(1, 26)}
    assert rep["pilot_results_sha256"] == F.sha256_file(str(res))


def test_EXACT_COLLAPSE_all_16_identical_tuples_is_STOP_COLLAPSE_with_the_fixed_wording(
        repo, templates):
    man, res = make_pilot(repo, templates, picks=[("red", "black")] * 16)
    rep = feasibility(man, res)
    assert rep["verdict"] == "STOP_COLLAPSE"
    (fired,) = rep["rules_fired"]
    assert fired["statement"] == "empirically complete concentration in the pilot"
    assert (fired["identical_pair_tuples"], fired["of"]) == (16, 16)
    assert "entropy" not in json.dumps(rep).lower(), "16 draws establish no zero entropy"


def test_FIFTEEN_identical_and_ONE_different_is_NOT_a_stop(repo, templates):
    rep = feasibility(*make_pilot(repo, templates))            # PROCEED_PICKS: 15 + 1
    assert rep["concentration"]["pair_tuples"]["max_frequency"] == 15
    assert rep["verdict"] == "PROCEED"


@pytest.mark.parametrize("total_s,verdict", [(2390.0, "PROCEED"), (2391.0, "STOP_RUNTIME")])
def test_RUNTIME_boundary_exactly_10800_proceeds_just_above_stops(repo, templates, total_s,
                                                                   verdict):
    rep = feasibility(*make_pilot(repo, templates, total_s=total_s))
    assert rep["verdict"] == verdict
    assert (rep["runtime"]["projected_segment_s"] > 10_800) == (verdict != "PROCEED")


@pytest.mark.parametrize("n_cap,verdict", [(3, "PROCEED"), (4, "STOP_CAP")])
def test_CAP_boundary_3_pairs_proceed_4_stop(repo, templates, n_cap, verdict):
    capped = [("cap", "black"), ("red", "cap"), ("cap", "cap"), ("cap", "black")][:n_cap]
    picks = [("red2", "black")] + capped + [("red", "black")] * (15 - n_cap)
    rep = feasibility(*make_pilot(repo, templates, picks=picks))
    assert rep["caps"]["cap_affected_pairs"] == n_cap
    assert rep["verdict"] == verdict


def test_PRECEDENCE_names_the_headline_and_LISTS_every_rule_that_fired(repo, templates):
    rep = feasibility(*make_pilot(repo, templates, picks=[("cap", "cap")] * 16,
                                  total_s=2391.0))
    assert rep["verdict"] == "STOP_RUNTIME"
    assert [f["rule"] for f in rep["rules_fired"]] == ["STOP_RUNTIME", "STOP_CAP",
                                                       "STOP_COLLAPSE"]


def test_the_precedence_is_VOID_RUNTIME_CAP_COLLAPSE():
    assert F.verdict([{"rule": r} for r in ("STOP_COLLAPSE", "STOP_CAP")]) == "STOP_CAP"
    assert F.verdict([{"rule": r} for r in ("STOP_CAP", "VOID")]) == "VOID"
    assert F.verdict([]) == "PROCEED"


# ─────────────────────── 2. VOID is reported, not refused (§1, §1.4) ───────────────────────

def _to_run_void(i, r):
    if r["record_type"] == "segment_end":
        return {"record_type": "run_void", "stage": "game", "task_id": "x", "pair_id": "x",
                "arm": "A", "ply": 3, "classification": "timeout",
                "exception": "TimeoutExpired", "reason": "staged", "stdout": None,
                "completed": {}, "elapsed_s": 1.0}
    return r


def _retarget_winner(i, r):
    """Flip one recorded winner AND recompute its transcript digest, so only the
    reader's replay of the moves can see it -- and its message names a winner."""
    if r["record_type"] == "game_result" and r["task_id"].endswith("p3-A"):
        r = dict(r, winner="black")
        plies = [p for p in _retarget_winner.plies if p["task_id"] == r["task_id"]]
        r["transcript_digest"] = R.transcript_digest_of(plies, r)
    elif r["record_type"] == "ply":
        _retarget_winner.plies.append(r)
    return r


@pytest.mark.parametrize("edit,why", [
    (_to_run_void, "run_void"),
    (lambda i, r: None if r["record_type"] == "segment_end" else r, "killed"),
    (lambda i, r: dict(r, transcript_digest="0" * 64)
     if r["record_type"] == "game_result" and i > 400 else r, "digest"),
    (lambda i, r: dict(r, complete=False) if r["record_type"] == "segment_end" else r,
     "incomplete")])
def test_an_OPERATIONAL_FAILURE_is_reported_VOID_not_refused(repo, templates, edit, why):
    rep = feasibility(*make_pilot(repo, templates, edit=edit))
    assert rep["verdict"] == "VOID"
    (fired,) = rep["rules_fired"]
    assert fired["rule"] == "VOID"
    assert set(rep) == {"report", "design", "pilot_results_sha256", "pilot_manifest_sha256",
                        "analysis_card_sha256", "verdict", "rules_fired", "written_at"}, \
        "a VOID pilot has NO feasibility result"
    if why == "run_void":
        assert fired["run_void"] == {"stage": "game", "classification": "timeout",
                                     "exception": "TimeoutExpired"}


def test_a_VOID_reason_that_NAMES_a_winner_is_withheld(repo, templates):
    _retarget_winner.plies = []
    rep = feasibility(*make_pilot(repo, templates, edit=_retarget_winner))
    assert rep["verdict"] == "VOID"
    reason = rep["rules_fired"][0]["reader"]
    assert "withheld" in reason and "red" not in reason and "black" not in reason


# ──────────────────────────── 3. wrong-but-valid inputs (§0.1) ────────────────────────────

def _header(fn):
    def edit(i, r):
        if i == 0:
            r = dict(r)
            fn(r)
        return r
    return edit


@pytest.mark.parametrize("mutate,match", [
    (lambda h: h.update(schedule_digest="0" * 64), "header `schedule_digest` differs"),
    (lambda h: h.update(seeds=h["seeds"][::-1]), "header `seeds` differs"),
    (lambda h: h.update(incumbent_identity={"stub": False}),
     "header `incumbent_identity` differs"),
    (lambda h: h.update(t1j_runtime=dict(h["t1j_runtime"], ply_cap=274)),
     "header `t1j_runtime` differs"),
    (lambda h: h.update(t1j_runtime=dict(h["t1j_runtime"], query_timeout_s=60)),
     "header `t1j_runtime` differs"),
    (lambda h: h.update(t1j_runtime=dict(h["t1j_runtime"], replay_timeout_s=60)),
     "header `t1j_runtime` differs"),
    (lambda h: h.update(cards=dict(h["cards"], **{F.CARD: "e" * 64})),
     "header `cards` differs"),
    (lambda h: h.update(code=dict(h["code"], **{F.MANIFEST_CODE[0]: "e" * 64})),
     "header `code` differs"),
    (lambda h: h.pop("code"), "no `code` field"),
    (lambda h: h.update(design=F.STUDY), "design 'H4_STUDY'"),
    (lambda h: h.update(segment=1), "segment 1 is not manifest slot 0"),
    (lambda h: h.update(evidence=True), "evidence flag True")])
def test_a_HEADER_that_differs_from_the_manifest_is_REFUSED_and_nothing_written(
        repo, templates, mutate, match):
    man, res = make_pilot(repo, templates, edit=_header(mutate))
    before = sorted(os.listdir(res.parent))
    with pytest.raises(F.H4AnalysisRefused, match=match):
        feasibility(man, res)
    assert sorted(os.listdir(res.parent)) == before, "a refusal writes nothing, no temp"


def test_a_pilot_from_ANOTHER_SCHEDULE_is_refused(repo, templates):
    other = entry(0, 16, tag="elsewhere", seed0=500)
    with pytest.raises(F.H4AnalysisRefused, match="schedule_digest"):
        feasibility(*make_pilot(repo, templates, manifest_entry=other))


@pytest.mark.parametrize("fix,match", [
    (lambda e: {"stage": F.STUDY, "segments": [e]}, "H4_PILOT manifest is required"),
    (lambda e: {"stage": F.PILOT, "segments": [e, e]}, "exactly 1 segment"),
    (lambda e: {"stage": F.PILOT, "segments": [entry(0, 15, tag="pilot")]}, "30 tasks"),
    (lambda e: {"stage": F.PILOT, "segments": [dict(e, schedule_digest="0" * 64)]},
     "does not recompute"),
    (lambda e: {"stage": F.PILOT, "segments": [dict(e, seeds=e["seeds"][::-1])]},
     "not the schedule's seeds"),
    (lambda e: {"stage": F.PILOT, "segments": [dict(e, cards={
        c: "c" * 64 for c in F.MANIFEST_CARDS[1:]})]}, "cards must be"),
    (lambda e: {"stage": F.PILOT, "segments": [dict(e, t1j_runtime=dict(
        e["t1j_runtime"], ply_cap=274))]}, "frozen runtime"),
    (lambda e: {"stage": F.PILOT, "segments": [dict(e, segment=3)]}, "labelled segment"),
    (lambda e: {"stage": F.PILOT, "segments": [{k: v for k, v in e.items()
                                                 if k != "code"}]}, "lacks")])
def test_a_MANIFEST_that_is_not_a_valid_pilot_preregistration_is_refused(repo, templates,
                                                                        fix, match):
    with pytest.raises(F.H4AnalysisRefused, match=match):
        feasibility(*make_pilot(repo, templates, manifest=fix(entry(0, 16, tag="pilot"))))


def test_GAMES_that_are_not_the_manifest_schedule_are_refused(repo, templates):
    def reseed(i, r):
        if r["record_type"] == "game_start" and r["game_index"] == 5:
            r = dict(r, seed=-9999)
        return r
    with pytest.raises(F.H4AnalysisRefused, match="not the manifest's schedule"):
        feasibility(*make_pilot(repo, templates, edit=reseed))


# ─────────────────────────────── 4. durability (§2.1) ───────────────────────────────

@pytest.mark.parametrize("which", ["manifest", "results"])
@pytest.mark.parametrize("how", ["untracked", "modified", "staged_not_committed"])
def test_every_input_must_be_COMMITTED_and_UNMODIFIED(repo, templates, which, how):
    other = {"manifest": "results", "results": "manifest"}[which]
    man, res = make_pilot(repo, templates, to_commit=(
        ("manifest", "results") if how == "modified" else (other,)))
    target = {"manifest": man, "results": res}[which]
    if how == "modified":
        target.write_text(target.read_text() + "\n")
    elif how == "staged_not_committed":
        git(repo, "add", "--", str(target))
    with pytest.raises(F.H4AnalysisRefused, match="not durable"):
        feasibility(man, res)
    assert not (res.parent / "feasibility.json").exists()


def test_the_durability_check_passes_a_committed_unmodified_file(repo):
    """CLEAN BASELINE: a check tightened until nothing passes would satisfy the above."""
    f = repo / "x.json"
    f.write_text("{}")
    commit(repo, f)
    F.check_durable([str(f)], _fixture=True)


#: A file this work never edits, tracked at HEAD in THIS repository.
TRACKED_HERE = ROOT / "scripts" / "GPU" / "alphazero" / "gate_inventory.py"


def test_PRODUCTION_durability_passes_a_committed_file_IN_THIS_REPOSITORY():
    """CLEAN BASELINE for the production path: it is not refusing everything."""
    F.check_durable([str(TRACKED_HERE)], _fixture=False)


def test_PRODUCTION_durability_refuses_a_committed_file_in_ANOTHER_repository(repo):
    f = repo / "x.json"
    f.write_text("{}")
    commit(repo, f)
    with pytest.raises(F.H4AnalysisRefused, match="does not resolve into this repository"):
        F.check_durable([str(f)], _fixture=False)


def test_PRODUCTION_durability_resolves_SYMLINKS_before_judging(repo, tmp_path):
    f = repo / "x.json"
    f.write_text("{}")
    commit(repo, f)
    out = tmp_path / "out-link.json"
    out.symlink_to(f)                                          # resolves OUT of this repo
    with pytest.raises(F.H4AnalysisRefused, match="does not resolve into this repository"):
        F.check_durable([str(out)], _fixture=False)
    into = tmp_path / "in-link.py"
    into.symlink_to(TRACKED_HERE)                              # resolves INTO this repo
    F.check_durable([str(into)], _fixture=False)


def test_PRODUCTION_durability_refuses_an_untracked_path_in_this_repository():
    with pytest.raises(F.H4AnalysisRefused, match="not tracked"):
        F.check_durable([str(ROOT / "docs" / "superpowers" / "evidence" /
                             "no-such-h4-file.json")], _fixture=False)


# ──────────────────────────── 5. output semantics (§4.1) ────────────────────────────

def test_an_OCCUPIED_destination_is_refused_BEFORE_any_input_is_read(tmp_path):
    out = tmp_path / "feasibility.json"
    out.write_text("prior")
    with pytest.raises(F.H4AnalysisRefused, match="occupied"):
        F._write_report(str(tmp_path / "no-manifest"), str(tmp_path / "no-results"),
                        str(out), _fixture=True)
    assert out.read_text() == "prior"


def test_create_only_write_refuses_a_name_taken_at_LINK_time_and_leaves_no_temp(tmp_path):
    out = tmp_path / "x.json"
    out.write_text("prior")
    with pytest.raises(F.H4AnalysisRefused, match="occupied"):
        F.write_create_only(str(out), {"a": 1})
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")], \
        "a temporary file survived the refusal"
    assert os.listdir(tmp_path) == ["x.json"] and out.read_text() == "prior"


def test_create_only_write_leaves_only_the_output(tmp_path):
    F.write_create_only(str(tmp_path / "x.json"), {"a": 1})
    assert os.listdir(tmp_path) == ["x.json"]
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}


# ──────────────────────────── 6. the BLIND (§1.2, §4) ────────────────────────────

@pytest.mark.parametrize("key", ["winner", "points", "t1j_points", "incumbent_points",
                                 "score", "pair_score", "interval", "mean",
                                 "outcome_patterns"])
def test_a_report_carrying_an_OUTCOME_field_is_refused(key):
    with pytest.raises(F.H4AnalysisRefused, match="outcome fields"):
        F.check_report_blind({"verdict": "PROCEED", "concentration": {"x": [{key: 1}]}})


def test_a_report_whose_TEXT_names_an_outcome_is_refused():
    """The runner's own text check, reused: a value, not only a key."""
    with pytest.raises(F.H4AnalysisRefused, match="t1j_points"):
        F.check_report_blind({"verdict": "PROCEED", "note": "t1j_points=1"})


def test_a_leaking_report_is_refused_BEFORE_it_is_written(repo, templates, monkeypatch):
    man, res = make_pilot(repo, templates)
    real = F.build_report
    monkeypatch.setattr(F, "build_report",
                        lambda *a, **k: dict(real(*a, **k), pair_score=0.75))
    with pytest.raises(F.H4AnalysisRefused, match="outcome fields"):
        feasibility(man, res)
    assert not (res.parent / "feasibility.json").exists()


def test_the_BLINDED_VIEW_carries_no_outcome_field_and_a_leaking_view_is_refused(
        repo, templates):
    _man, res = make_pilot(repo, templates)
    view = F.blinded_view(R.load_games(str(res), allow_fixture=True))
    assert len(view) == 32
    assert not {k for v in view for k in v} & set(R.BLIND_FIELDS)
    with pytest.raises(F.H4AnalysisRefused, match="blinded view carries"):
        F.check_view([dict(view[0], winner="red")])


BLIND_CONSTANTS = {"winner", "points", "t1j_points", "incumbent_points", "score",
                   "pair_score"}
FORBIDDEN_IMPORTS = {"h4_pilot_decision", "h4_confirmatory_analysis"}


def blind_reads(source):
    """Executable code only (docstrings skipped): an outcome field named, a
    `.winner`, or an import of the decision or confirmatory module."""
    tree = ast.parse(source)
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)}
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and id(n) not in docs and n.value in BLIND_CONSTANTS:
            found.append(f"constant {n.value!r}")
        elif isinstance(n, ast.Attribute) and n.attr == "winner":
            found.append(".winner")
        elif isinstance(n, ast.Name) and n.id == "winner":
            found.append("name winner")
        elif isinstance(n, ast.ImportFrom):
            names = {a.name for a in n.names} | {n.module or ""}
            found += [f"import {x}" for x in names if x.split(".")[-1] in FORBIDDEN_IMPORTS]
        elif isinstance(n, ast.Import):
            found += [f"import {a.name}" for a in n.names
                      if a.name.split(".")[-1] in FORBIDDEN_IMPORTS]
    return found


def test_the_FEASIBILITY_MODULE_reads_no_outcome_and_imports_neither_other_module():
    assert blind_reads(pathlib.Path(F.__file__).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("planted", [
    'x = g["result"]["winner"]', "y = final.winner()", 'z = r.get("t1j_points")',
    "from . import h4_confirmatory_analysis", "from . import h4_pilot_decision as D",
    "import scripts.GPU.alphazero.h4_confirmatory_analysis"])
def test_the_blind_walker_is_NOT_VACUOUS(planted):
    """CLEAN-BASELINE CONTROL: the walker sees a planted read in the real source."""
    src = pathlib.Path(F.__file__).read_text(encoding="utf-8") + "\n" + planted + "\n"
    assert blind_reads(src) != []


# ───────────────── 7. the fixture relaxation is unreachable (§4) ─────────────────

def test_the_PUBLIC_entry_refuses_a_committed_FIXTURE(repo, templates):
    """Refused where it lives: a temporary repository is not this one."""
    man, res = make_pilot(repo, templates)
    with pytest.raises(F.H4AnalysisRefused, match="does not resolve into this repository"):
        F.write_feasibility_report(str(man), str(res), str(res.parent / "f.json"))
    assert not (res.parent / "f.json").exists()


def test_the_PUBLIC_entry_refuses_FIXTURE_CONTENT_even_inside_the_repository(
        repo, templates, monkeypatch):
    """With the location check satisfied -- the repository root pointed at the
    fixture's own repo, for this test only -- the content is still refused, at
    the manifest's negative seeds."""
    man, res = make_pilot(repo, templates)
    monkeypatch.setattr(R, "REPO_ROOT", repo.resolve())
    with pytest.raises(F.H4AnalysisRefused, match="negative seed"):
        F.write_feasibility_report(str(man), str(res), str(res.parent / "f.json"))
    assert not (res.parent / "f.json").exists()


def test_the_CLI_refuses_a_committed_FIXTURE_as_a_fresh_subprocess(repo, templates):
    man, res = make_pilot(repo, templates)
    out = res.parent / "f.json"
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h4_pilot_feasibility",
                        "--manifest", str(man), "--results", str(res), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == F.EXIT_REFUSED, f"exit {r.returncode}, stderr {r.stderr!r}"
    assert "does not resolve into this repository" in r.stderr and not out.exists()


def fixture_leaks(module_path, public):
    """A public entry with a `_fixture` parameter, or passing anything but a
    literal False for it."""
    tree = ast.parse(pathlib.Path(module_path).read_text(encoding="utf-8"))
    bad = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name in public):
        args = fn.args.args + fn.args.kwonlyargs
        if any(a.arg == "_fixture" for a in args) or fn.args.kwarg:
            bad.append(f"{fn.name} accepts _fixture")
        for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
            for kw in call.keywords:
                if kw.arg == "_fixture" and not (isinstance(kw.value, ast.Constant)
                                                 and kw.value.value is False):
                    bad.append(f"{fn.name} passes _fixture={ast.unparse(kw.value)}")
    return bad


def test_the_MANIFEST_names_exactly_the_cards_and_code_the_card_names():
    """Spelled out here from card §0.1, not read from the module under test."""
    assert list(F.MANIFEST_CARDS) == [
        "docs/superpowers/2026-09-22-t1j-h4-runner-persistence-card.md",
        "docs/superpowers/2026-09-22-t1j-h4-4b-acceptance-qualification-card.md",
        "docs/superpowers/2026-09-21-t1j-h4-replacement-card.md",
        "docs/superpowers/2026-09-23-t1j-h4-analysis-card.md"]
    # `code` is the runner's reviewed play-path list (runner card §12.6.1), read,
    # never retyped; `tests/test_h4_production_path.py` pins it to the card.
    assert F.MANIFEST_CODE is R.CODE


def test_NO_public_entry_can_reach_the_fixture_relaxation():
    assert fixture_leaks(F.__file__, {"write_feasibility_report", "main"}) == []


def test_the_fixture_walker_is_NOT_VACUOUS(tmp_path):
    """CLEAN-BASELINE CONTROL: a public entry passing _fixture=True is seen."""
    src = pathlib.Path(F.__file__).read_text(encoding="utf-8").replace(
        "return _main(argv, _fixture=False)", "return _main(argv, _fixture=True)")
    p = tmp_path / "planted.py"
    p.write_text(src)
    assert fixture_leaks(p, {"main"}) == ["main passes _fixture=True"]


def test_the_CLI_writes_the_report_and_exits_0_through_the_fixture_core(repo, templates):
    man, res = make_pilot(repo, templates)
    out = res.parent / "f.json"
    assert F._main(["--manifest", str(man), "--results", str(res), "--out", str(out)],
                   _fixture=True) == F.EXIT_OK
    assert json.loads(out.read_text())["verdict"] == "PROCEED"
