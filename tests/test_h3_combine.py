"""H3 FULL STUDY — THE COMBINATION, exercised end to end on SYNTHETIC records.

🔴 NOTHING HERE TOUCHES THE REAL RESULTS. Every test builds its own four segment
directories in a temp path and writes to a temp destination. The combination is
a one-shot, create-only act against a frozen destination; a test that reached it
would spend it, and the four runs behind it cannot be repeated.

🔑 THE SEEDS AND BLOCKS ARE REAL, THE GAMES ARE NOT. `verify_final_state` reads
the SEED REGISTRY, and the registry knows only the real blocks. Stubbing the
verifier out would be a permissive double standing in for the exact seam these
tests exist to cover -- the defect that let a whole pilot schedule through with
no reference fields. So the synthetic records carry the real seeds, and every
outcome, digest and transcript is fabricated.
"""
import hashlib
import json
import os
import pathlib

import pytest

from scripts.GPU.alphazero import h3_combine as COMBINE
from scripts.GPU.alphazero import h3_study_analysis as ANALYSIS
from scripts.GPU.alphazero import h3_study_rules as R
from scripts.GPU.alphazero import h3_study_runner as RUN

PAIRS_PER_SEGMENT = R.GAMES_PER_SEGMENT // 2


def _digest(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _segment_records(k, block, *, winner_of=None):
    """148 fabricated games on segment k's REAL seeds, in 74 complete pairs."""
    lo, hi = block
    seeds = list(range(lo, hi))
    rows = [{"record_type": "header", "design": "H3_FULL_STUDY", "segment": k,
             "identity": {}, "selection_mode": "frozen",
             "opening_set_digest": R.OPENING_SET_DIGEST}]
    results = []
    for i in range(PAIRS_PER_SEGMENT):
        pair_id = k * PAIRS_PER_SEGMENT + i
        opening = _digest("opening", pair_id)
        for arm, colour in enumerate(("red", "black")):
            seed = seeds[i * 2 + arm]
            win = (winner_of or (lambda p, a: "red"))(pair_id, arm)
            results.append({
                "record_type": "task_result",
                "task_id": f"h3study-{seed}-p{pair_id}-inc_{colour}",
                "pair_id": pair_id, "seed": seed, "segment": k,
                "stratum": R.STRATUM_UNIFORM,
                "opening_digest": opening,
                "transcript_digest": _digest("transcript", seed),
                "incumbent_colour": colour, "winner": win,
                "terminal_reason": "win", "plies": 60, "elapsed_s": 40.0,
            })
    for r in results:
        rows.append({"record_type": "transcript", "task_id": r["task_id"],
                     "moves": [], "transcript_digest": r["transcript_digest"]})
    rows.extend(results)
    return rows, results


def build_segment(root, k, block, *, winner_of=None, drop_game=False,
                  duplicate_game=False, split_pair=False):
    """One synthetic segment directory that `verify_final_state` accepts."""
    d = pathlib.Path(root) / f"segment{k}"
    d.mkdir(parents=True, exist_ok=True)
    rows, results = _segment_records(k, block, winner_of=winner_of)
    if drop_game:
        rows = [r for r in rows if r is not results[-1]]
        results = results[:-1]
    if split_pair:
        #: 592 games and 592 seeds still, but one pair holds THREE and another
        #: holds ONE -- the shape a count-only check cannot see.
        results[-1] = dict(results[-1], pair_id=results[0]["pair_id"])
        rows = [dict(r, pair_id=results[0]["pair_id"])
                if r.get("task_id") == results[-1]["task_id"]
                and r.get("record_type") == "task_result" else r for r in rows]
    n = len(results)
    report_from = list(results)
    if duplicate_game:
        #: the RESULTS FILE carries the repeat; the report is still built from
        #: the clean set, because `summarise` refuses a duplicate task_id and a
        #: fixture that could not build its report would be testing nothing.
        rows.append(dict(results[0]))
    (d / "03_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    trace = [{"event": "run_start", "segment": k}]
    for r in results:
        trace.append({"event": "task_start", "task_id": r["task_id"]})
        trace.append({"event": "task_done", "task_id": r["task_id"]})
    trace.append({"event": "run_end", "segment": k, "verdict": "OK",
                  "games_completed": n})
    (d / "04_trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in trace) + "\n")

    report = ANALYSIS.summarise(report_from, total_elapsed_s=6000.0)
    #: the three fields the RUNNER adds after summarise -- the real reports
    #: carry them and `verify_final_state` requires them, so a fixture without
    #: them would be a record the real path never produces.
    report.update({"segment": k, "complete": True, "timed_out": False})
    (d / "09_report.json").write_text(json.dumps(report))
    (d / "00_launch_receipt.json").write_text(json.dumps({
        "design": "H3_FULL_STUDY_SEGMENT", "segment": k, "outcome": "COMPLETED",
        "exit_code": 0, "worker_exit": 0, "gate_restored": True,
        "gate_readback": "False", "group_cleared": True, "interrupted": False,
        "timed_out": False, "segment_digest": RUN.SEGMENT_DIGESTS[k],
    }))
    return d


@pytest.fixture
def study(tmp_path):
    """Four synthetic segments on the real blocks, plus accessors for them."""
    src = tmp_path / "segments"
    for k, block in enumerate(RUN.SEGMENT_SEED_BLOCKS):
        build_segment(src, k, block)

    def paths_for(k):
        d = src / f"segment{k}"
        return (str(d / "03_results.jsonl"), str(d / "04_trace.jsonl"),
                str(d / "09_report.json"))

    def receipt_for(k):
        return str(src / f"segment{k}" / "00_launch_receipt.json")

    return {"root": src, "paths_for": paths_for, "receipt_for": receipt_for,
            "out": str(tmp_path / "combined")}


def run(study, **kw):
    return COMBINE.combine_unguarded(
        out_dir=kw.pop("out_dir", study["out"]),
        paths_for=kw.pop("paths_for", study["paths_for"]),
        receipt_for=kw.pop("receipt_for", study["receipt_for"]), **kw)


# ═══════════════════════════ the gate ═══════════════════════════════════════
def test_THE_COMBINATION_GATE_IS_SHUT_AND_THE_DESTINATION_IS_ABSENT():
    assert COMBINE.H3_COMBINATION_AUTHORIZED is False
    assert not os.path.lexists(COMBINE.COMBINED_OUT_DIR)
    assert not os.path.lexists(COMBINE.COMBINED_REPORT)


def test_combine_REFUSES_WHILE_THE_GATE_IS_SHUT():
    with pytest.raises(COMBINE.H3CombineError, match="NOT AUTHORIZED"):
        COMBINE.combine()
    assert not os.path.lexists(COMBINE.COMBINED_OUT_DIR), (
        "a refusal must not create the destination")


def test_combine_TAKES_NO_ARGUMENTS_AT_ALL():
    """🔴 AN ENTRY POINT THAT ACCEPTS ITS OWN SEAMS IS NOT A GATE. If `combine()`
    could be handed other records, another destination, a looser verifier or a
    different analysis, then opening the gate would authorize something other
    than the thing reviewed."""
    import inspect
    assert list(inspect.signature(COMBINE.combine).parameters) == []


def test_THE_FROZEN_ENTRY_POINT_USES_THE_REAL_COLLABORATORS():
    """Defaults are asserted BY IDENTITY, not by equality: two functions that
    behave alike are not the same object, and the record must describe the one
    that ran."""
    from scripts.GPU.alphazero import h3_final_state as FINAL
    import inspect
    src = inspect.getsource(COMBINE.combine)
    assert "combine_unguarded(out_dir=COMBINED_OUT_DIR)" in src
    assert COMBINE.FINAL.verify_final_state is FINAL.verify_final_state
    assert COMBINE.ANALYSIS.summarise is ANALYSIS.summarise
    assert COMBINE.RUN.combine_segments is RUN.combine_segments


# ═══════════════════════════ end to end ═════════════════════════════════════
def test_THE_WHOLE_PATH_RUNS_ON_SYNTHETIC_RECORDS(study):
    out = run(study)
    assert out["design"] == "H3_FULL_STUDY_COMBINED"
    assert out["n_games"] == R.N_GAMES == 592
    assert out["n_pairs"] == R.N_PAIRS == 296
    assert out["pairs_scored"] == 296
    assert out["below_report_floor"] is False, (
        "296 pairs clears the 148-pair floor")
    for key in ("primary", "sensitivities", "gates", "verdict", "verdict_note",
                "is_strength_verdict", "interpretation_withheld",
                "duplicate_pairs", "within_pair_identical", "capped_games"):
        assert key in out, key
    assert set(out["sensitivities"]) == set(COMBINE.REQUIRED_SENSITIVITIES)
    assert len(out["sensitivities"]) == 3

    written = json.loads(pathlib.Path(
        study["out"], os.path.basename(COMBINE.COMBINED_REPORT)).read_text())
    assert written == out, "the returned payload IS the persisted one"


def test_EVERY_INPUT_IS_RECORDED_BY_PATH_AND_HASH(study):
    out = run(study)
    assert len(out["inputs"]) == 4
    for k, entry in enumerate(out["inputs"]):
        assert entry["segment"] == k
        assert entry["games"] == R.GAMES_PER_SEGMENT
        assert entry["seed_block"] == list(RUN.SEGMENT_SEED_BLOCKS[k])
        for field in ("results", "trace", "report", "receipt"):
            path = entry[f"{field}_path"]
            fresh = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
            assert entry[f"{field}_sha256"] == fresh, field


def test_summarise_IS_CALLED_EXACTLY_ONCE(study):
    """🔴 ONCE, ON THE POOLED GAMES. Calling it per segment and averaging would
    be a different estimator wearing the preregistered one's name."""
    calls = []

    def spy(games, **kw):
        calls.append(len(games))
        return ANALYSIS.summarise(games, **kw)

    run(study, summarise=spy)
    assert calls == [R.N_GAMES], calls


def test_THE_POOLED_ELAPSED_TIME_IS_THE_SUM_OF_THE_SEGMENTS(study):
    out = run(study)
    assert out["total_elapsed_s"] == pytest.approx(4 * 6000.0)


# ═══════════════════════ what it must refuse ════════════════════════════════
def test_A_FINAL_STATE_DISAGREEMENT_BLOCKS_AGGREGATION(study):
    """Zero disagreements first, or the number describes no run."""
    receipt = pathlib.Path(study["receipt_for"](1))
    data = json.loads(receipt.read_text())
    data["outcome"] = "TIMED_OUT"
    receipt.write_text(json.dumps(data))
    with pytest.raises(COMBINE.H3CombineError, match="disagreement"):
        run(study)
    assert not os.path.lexists(study["out"]), "a refusal writes nothing"


def _broken(tmp_path, **flags):
    """Four segments with one deliberately malformed, and the combiner's OWN
    count guards isolated.

    🔴 THE PERMISSIVE `verify` IS CONFINED TO THESE TWO TESTS AND IS THE POINT
    OF THEM. `verify_final_state` catches a dropped or repeated game first, so
    with the real verifier these would pass for the wrong reason and the
    combiner's own 592/296 arithmetic would never be exercised -- defence in
    depth that nothing depends on is not defence. Every other test here uses the
    real verifier.
    """
    src = tmp_path / "segments"
    for k, block in enumerate(RUN.SEGMENT_SEED_BLOCKS):
        build_segment(src, k, block, **{f: (k == flags[f]) for f in flags})
    return dict(
        out_dir=str(tmp_path / "out"),
        paths_for=lambda k: (str(src / f"segment{k}" / "03_results.jsonl"),
                             str(src / f"segment{k}" / "04_trace.jsonl"),
                             str(src / f"segment{k}" / "09_report.json")),
        receipt_for=lambda k: str(src / f"segment{k}" / "00_launch_receipt.json"),
        verify=lambda **kw: [])


def test_A_MISSING_GAME_IS_REFUSED(tmp_path):
    """591 games is not the study, and the shortfall must not be averaged over."""
    with pytest.raises(COMBINE.H3CombineError, match="591 games"):
        COMBINE.combine_unguarded(**_broken(tmp_path, drop_game=2))


def test_A_DUPLICATED_GAME_IS_REFUSED(tmp_path):
    """The same game twice is a harness fault, not a result."""
    with pytest.raises(COMBINE.H3CombineError, match="distinct task ids"):
        COMBINE.combine_unguarded(**_broken(tmp_path, duplicate_game=0))


def test_AN_INCOMPLETE_PAIR_IS_REFUSED(tmp_path):
    """🔴 592 GAMES IS NOT 296 PAIRS. A pair holding three games and another
    holding one passes every count and is still not the design: the pair is the
    unit of observation, and a half pair contributes no within-pair comparison.
    """
    with pytest.raises(COMBINE.H3CombineError, match="exactly two games"):
        COMBINE.combine_unguarded(**_broken(tmp_path, split_pair=0))


def test_THE_REAL_VERIFIER_CATCHES_BOTH_FIRST(tmp_path):
    """And with the real verifier in place they never reach that arithmetic --
    which is the composition working, not the guard being redundant."""
    kw = _broken(tmp_path, drop_game=2)
    kw.pop("verify")
    with pytest.raises(COMBINE.H3CombineError, match="disagreement"):
        COMBINE.combine_unguarded(**kw)


def test_A_DUPLICATED_SEGMENT_IS_REFUSED(study):
    """Reading segment 1 twice would double 148 games and halve nothing."""
    with pytest.raises(COMBINE.H3CombineError):
        run(study, paths_for=lambda k: study["paths_for"](1 if k else 0))


def test_FEWER_THAN_FOUR_SEGMENTS_IS_REFUSED(study):
    with pytest.raises(COMBINE.H3CombineError, match="seed blocks"):
        run(study, blocks=RUN.SEGMENT_SEED_BLOCKS[:3])


def test_A_REFUSED_PERMISSION_STOPS_THE_COMBINATION(study):
    """🔴 PROVENANCE, NOT COUNTS. 296 pairs cannot buy what optional stopping
    forbids, and no arithmetic here may substitute for combine_segments."""
    with pytest.raises(COMBINE.H3CombineError, match="REFUSES a verdict"):
        run(study, permit=lambda rows: {"verdict_permitted": False,
                                        "flagged": True,
                                        "why": "a segment was WITHHELD"})
    assert not os.path.lexists(study["out"])


def test_AN_OMITTED_SENSITIVITY_IS_REFUSED(study):
    """Persisting a narrower analysis than the frozen one is not an option."""
    def narrowed(games, **kw):
        rep = ANALYSIS.summarise(games, **kw)
        rep["sensitivities"] = {k: v for k, v in rep["sensitivities"].items()
                                if k != "both"}
        return rep
    with pytest.raises(COMBINE.H3CombineError, match="omits the preregistered"):
        run(study, summarise=narrowed)


# ═══════════════════════ the destination ════════════════════════════════════
def test_THE_DESTINATION_IS_CREATE_ONLY(study):
    run(study)
    with pytest.raises(COMBINE.H3CombineError, match="already exists"):
        run(study)


def test_A_DANGLING_SYMLINK_ALSO_OCCUPIES_THE_DESTINATION(study, tmp_path):
    """`O_EXCL` follows a symlink; `lexists` does not."""
    out = pathlib.Path(study["out"])
    out.mkdir(parents=True)
    (out / os.path.basename(COMBINE.COMBINED_REPORT)).symlink_to(
        tmp_path / "nowhere")
    with pytest.raises(COMBINE.H3CombineError, match="already exists"):
        run(study)


def test_A_PARTIAL_WRITE_LEAVES_NO_FILE_BEHIND(study, monkeypatch):
    """🔴 A TRUNCATED REPORT WOULD OCCUPY THE FROZEN DESTINATION FOREVER, and
    the next attempt would be refused for a file that never held a result."""
    real_dump = json.dump

    def explode(obj, fh, **kw):
        fh.write('{"partial": ')
        raise OSError("disk full, half-way through")

    monkeypatch.setattr(COMBINE.json, "dump", explode)
    with pytest.raises(OSError):
        run(study)
    path = pathlib.Path(study["out"], os.path.basename(COMBINE.COMBINED_REPORT))
    assert not os.path.lexists(path), "the partial file must be removed"

    monkeypatch.setattr(COMBINE.json, "dump", real_dump)
    out = run(study)                       # and the destination is free again
    assert out["n_games"] == R.N_GAMES
