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
                  duplicate_game=False, split_pair=False,
                  shift_seeds=False):
    """One synthetic segment directory that `verify_final_state` accepts."""
    d = pathlib.Path(root) / f"segment{k}"
    d.mkdir(parents=True, exist_ok=True)
    rows, results = _segment_records(k, block, winner_of=winner_of)
    if drop_game:
        rows = [r for r in rows if r is not results[-1]]
        results = results[:-1]
    if shift_seeds:
        #: distinct task ids, complete pairs, 592 games -- and seeds that are
        #: NOT the block this segment was allocated.
        results = [dict(r, seed=r["seed"] + 10_000_000) for r in results]
        rows = [dict(r, seed=r["seed"] + 10_000_000)
                if r.get("record_type") == "task_result" else r for r in rows]
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
def test_THE_COMBINATION_HAS_RUN_AND_ITS_GATE_IS_SHUT_AGAIN():
    """🔴 INVERTED AFTER THE COMBINATION RAN (2026-09-20, attempt 2). The
    destination is no longer absent -- it holds the study's answer -- and the
    claim that survives is that the gate is shut and the outputs are the ones
    that were produced. Their digests are re-hashed, not read back."""
    import hashlib
    assert COMBINE.H3_COMBINATION_AUTHORIZED is False
    assert CCMD.gate_readback() == "False"
    for path, pin in ((COMBINE.COMBINED_REPORT, COMBINE.COMBINED_REPORT_DIGEST),
                      (CCMD.RECEIPT, COMBINE.COMBINATION_RECEIPT_DIGEST)):
        assert os.path.lexists(path), path
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == pin, (
            "the combined output must still be byte-identical to what was written")


def test_combine_REFUSES_WHILE_THE_GATE_IS_SHUT():
    before = sorted(pathlib.Path(COMBINE.COMBINED_OUT_DIR).iterdir())
    with pytest.raises(COMBINE.H3CombineError, match="NOT AUTHORIZED"):
        COMBINE.combine()
    assert sorted(pathlib.Path(COMBINE.COMBINED_OUT_DIR).iterdir()) == before, (
        "a refusal must not touch the destination -- which now HOLDS the result, "
        "so 'created nothing' is no longer the whole claim")


def test_THE_COMBINATION_CANNOT_BE_RUN_A_SECOND_TIME(monkeypatch):
    """One shot. Even with the gate open, the destination is occupied and the
    create-only install refuses -- the report that exists is the only one."""
    monkeypatch.setattr(COMBINE, "H3_COMBINATION_AUTHORIZED", True)
    from scripts.GPU.alphazero import gate_inventory as INV
    monkeypatch.setattr(INV, "open_gates",
                        lambda: [("h3_combine", "H3_COMBINATION_AUTHORIZED")])
    with pytest.raises(COMBINE.H3CombineError, match="already exists"):
        COMBINE.combine()


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
            assert entry[f"{field}_sha256"] == fresh, (
                "every input must be recorded with the hash of what was read")


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
    #: 🔴 THE TRACE, NOT THE RECEIPT. Tampering the receipt's outcome ALSO
    #: turns the segment "void" for `combine_segments`, so removing the
    #: final-state check moved the refusal rather than removing it and the
    #: control never demonstrated its own claim. A trace that did not end OK is
    #: a disagreement the final-state check alone can see.
    trace = pathlib.Path(study["paths_for"](1)[1])
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    rows[-1]["verdict"] = "VOID"
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
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
    """Reading segment 1 three times would triple its 148 games and halve
    nothing. The FINAL-STATE check sees it first -- segment 1's trace and report
    name segment 1 wherever they are read from -- and the test names which guard
    fired, because an assertion satisfied by any refusal at all proves nothing
    about which one is doing the work. The task-id arithmetic behind it is
    exercised in isolation by test_A_DUPLICATED_GAME_IS_REFUSED."""
    with pytest.raises(COMBINE.H3CombineError, match="disagreement"):
        run(study, paths_for=lambda k: study["paths_for"](1 if k else 0))


def test_SEEDS_THAT_ARE_NOT_THE_FOUR_BLOCKS_ARE_REFUSED(tmp_path):
    """🔴 THE GAMES MUST BE THE GAMES THAT WERE SCHEDULED. Distinct task ids,
    complete pairs and 592 records are all satisfiable by games played on other
    seeds entirely; only the union check says they are THIS study's."""
    with pytest.raises(COMBINE.H3CombineError, match="union of the four blocks"):
        COMBINE.combine_unguarded(**_broken(tmp_path, shift_seeds=0))


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
        run(study)          # the EARLY check, named distinctly from the race


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


# ═══════════════ the official path must appear ONLY when durable ════════════
def test_THE_OFFICIAL_PATH_IS_NEVER_OPENED_FOR_WRITING(study, monkeypatch):
    """🔴 AN EXCEPTION HANDLER DOES NOT SURVIVE A KILL.

    The first version opened the FINAL path with O_EXCL and serialised into it,
    removing the partial file in an `except`. A Python exception was handled; a
    SIGKILL, a crash or a power loss was not, and any of them would leave a
    truncated report sitting at the frozen destination forever -- occupying the
    one-shot path with a file that never held a result.

    So nothing is ever written THROUGH the official path. It is created by
    `os.link` from a file that is already complete and fsynced.
    """
    opened = []
    real_open = os.open

    def spy(path, flags, *a, **kw):
        if isinstance(path, str) and "WRONLY" not in str(flags):
            pass
        if isinstance(path, str) and (flags & os.O_WRONLY or flags & os.O_RDWR):
            opened.append(path)
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(COMBINE.os, "open", spy)
    run(study)
    final = os.path.join(study["out"], os.path.basename(COMBINE.COMBINED_REPORT))
    assert opened, "the spy saw nothing; it is not wired to the writer"
    assert final not in opened, (
        f"the official path was opened for writing: {opened}")
    assert all(os.path.dirname(p) == os.path.dirname(final) for p in opened), (
        f"every write must land in the destination's own directory: {opened}")


def test_A_KILL_BEFORE_THE_LINK_LEAVES_NO_OFFICIAL_FILE(study, monkeypatch):
    """The window the handler could not cover: die after the payload is written
    but before it is installed. The official path must simply not exist."""
    def die(src, dst):
        raise KeyboardInterrupt("SIGINT between fsync and link")

    monkeypatch.setattr(COMBINE.os, "link", die)
    with pytest.raises(KeyboardInterrupt):
        run(study)
    final = pathlib.Path(study["out"], os.path.basename(COMBINE.COMBINED_REPORT))
    assert not os.path.lexists(final)


def test_THE_TEMPORARY_FILE_IS_IN_THE_SAME_DIRECTORY_AND_IS_REMOVED(study):
    """`os.link` cannot cross a filesystem, and a leftover temp file beside the
    report is litter at an evidence path."""
    seen = []
    real_open = COMBINE.os.open

    def spy(path, flags, *a, **kw):
        if isinstance(path, str) and (flags & os.O_CREAT):
            seen.append(path)
        return real_open(path, flags, *a, **kw)

    import unittest.mock as mock
    with mock.patch.object(COMBINE.os, "open", spy):
        run(study)
    final = pathlib.Path(study["out"], os.path.basename(COMBINE.COMBINED_REPORT))
    tmps = [p for p in seen if p != str(final)]
    assert tmps, "no temporary file was created"    # else nothing is proven
    for t in tmps:
        assert os.path.dirname(t) == str(final.parent), t
        assert not os.path.lexists(t), (
            "a temporary file was left beside the report")
    assert sorted(p.name for p in final.parent.iterdir()) == [final.name]


def test_THE_INSTALLED_REPORT_IS_COMPLETE_JSON(study):
    out = run(study)
    final = pathlib.Path(study["out"], os.path.basename(COMBINE.COMBINED_REPORT))
    assert json.loads(final.read_text()) == out


def test_THE_INSTALL_IS_CREATE_ONLY_NOT_A_RENAME(study):
    """🔴 `os.rename` WOULD SILENTLY OVERWRITE. The install must be the atomic
    create-only primitive, so a second combination cannot replace the first."""
    import inspect
    src = inspect.getsource(COMBINE._write_create_only)
    assert "os.link(" in src
    assert "os.rename(" not in src and "os.replace(" not in src


# ═════════════════ the supervised command and its restoration ═══════════════
from scripts.GPU.alphazero import h3_combine_command as CCMD   # noqa: E402

GATE_OPEN_SRC = "H3_COMBINATION_AUTHORIZED = True\n"
GATE_SHUT_SRC = "H3_COMBINATION_AUTHORIZED = False\n"


@pytest.fixture
def fake_source(tmp_path):
    """A stand-in for h3_combine.py's source, so restoration can be exercised
    without ever editing the real gate."""
    p = tmp_path / "h3_combine_copy.py"
    p.write_text("# header\n" + GATE_OPEN_SRC + "# tail\n")
    return str(p)


def test_THE_COMMAND_ACCEPTS_NOTHING_BUT_run():
    """🔴 NO PATH, NO INPUT, NO DESTINATION, NO COLLABORATOR. Anything that
    could aim this elsewhere would make opening the gate authorize something
    other than what was reviewed."""
    opts = CCMD._parser()._actions
    flags = sorted(f for a in opts for f in a.option_strings)
    assert flags == ["--help", "--run", "-h"], flags


def test_WITHOUT_run_IT_DOES_NOTHING(tmp_path):
    receipt = str(tmp_path / "receipt.json")
    assert CCMD.main([], _receipt=receipt) == CCMD.EXIT_REFUSED
    assert not os.path.lexists(receipt)


def test_A_SHUT_GATE_REFUSES_AND_WRITES_NO_RECEIPT(tmp_path, fake_source):
    """A receipt here would CREATE THE DESTINATION, and the real combination
    would then be refused for a directory occupied by a refusal."""
    pathlib.Path(fake_source).write_text("# header\n" + GATE_SHUT_SRC)
    receipt = str(tmp_path / "out" / "receipt.json")
    assert CCMD.main(["--run"], _combine_source=fake_source,
                     _receipt=receipt) == CCMD.EXIT_NOT_AUTHORIZED
    assert not os.path.lexists(receipt)
    assert not os.path.lexists(os.path.dirname(receipt))


def test_THE_GATE_IS_RESTORED_AND_READ_BACK_FROM_THE_SOURCE(fake_source):
    assert CCMD.gate_is_open(fake_source) is True
    assert CCMD.restore_gate(fake_source) is True
    assert CCMD.gate_readback(fake_source) == "False"
    assert CCMD.gate_is_open(fake_source) is False
    assert pathlib.Path(fake_source).read_text() == (
        "# header\n" + GATE_SHUT_SRC + "# tail\n")


def test_RESTORATION_IS_IDEMPOTENT_AND_HONEST_ABOUT_AN_UNREADABLE_SOURCE(tmp_path):
    shut = tmp_path / "already_shut.py"
    shut.write_text(GATE_SHUT_SRC)
    assert CCMD.restore_gate(str(shut)) is True
    missing = str(tmp_path / "gone.py")
    assert CCMD.restore_gate(missing) is False
    assert CCMD.gate_readback(missing) is None


def test_A_FAILING_COMBINATION_STILL_RESTORES_THE_GATE(tmp_path, fake_source,
                                                       monkeypatch):
    """🔴 THE CASE THAT MATTERS. An exception is exactly when nobody remembers
    to close a gate by hand."""
    def boom():
        raise RuntimeError("the combination exploded")

    monkeypatch.setattr(COMBINE, "combine", boom)
    receipt = str(tmp_path / "out" / "receipt.json")
    code = CCMD.main(["--run"], _combine_source=fake_source, _receipt=receipt)
    assert code == CCMD.EXIT_FAILED
    assert CCMD.gate_readback(fake_source) == "False"
    rec = json.loads(pathlib.Path(receipt).read_text())
    assert rec["outcome"] == "FAILED"
    assert rec["gate_restored"] is True and rec["gate_readback"] == "False"
    assert "the combination exploded" in rec["detail"]


def test_A_FAILED_RESTORATION_SUPERSEDES_A_SUCCESSFUL_COMBINATION(
        tmp_path, fake_source, monkeypatch):
    """🔴 SUPERSEDING. A tree left able to combine again is the more urgent
    fact about the run than the run having worked."""
    monkeypatch.setattr(COMBINE, "combine", lambda: {"n_games": 592,
                                                     "n_pairs": 296,
                                                     "verdict": "NO VERDICT"})
    monkeypatch.setattr(CCMD, "restore_gate", lambda *a, **k: False)
    receipt = str(tmp_path / "out" / "receipt.json")
    code = CCMD.main(["--run"], _combine_source=fake_source, _receipt=receipt)
    assert code == CCMD.EXIT_GATE_NOT_RESTORED
    rec = json.loads(pathlib.Path(receipt).read_text())
    assert rec["outcome"] == "COMPLETED"
    assert rec["gate_restored"] is False
    assert rec["exit_code"] == CCMD.EXIT_GATE_NOT_RESTORED, (
        "the receipt must record the superseding code, not the run's own")


def test_THE_RECEIPT_RECORDS_THE_INPUTS_AND_THE_OUTPUT(tmp_path, fake_source,
                                                       monkeypatch):
    monkeypatch.setattr(COMBINE, "combine", lambda: {
        "n_games": 592, "n_pairs": 296, "verdict": "NO VERDICT",
        "interpretation_withheld": False, "is_strength_verdict": True})
    #: 🔴 REDIRECTED, so `report_exists: False` means THIS run wrote nothing.
    #: The real report now exists, and against it the field would report the
    #: study's own answer rather than anything about the stubbed attempt.
    monkeypatch.setattr(COMBINE, "COMBINED_REPORT", str(tmp_path / "no-report.json"))
    receipt = str(tmp_path / "out" / "receipt.json")
    assert CCMD.main(["--run"], _combine_source=fake_source,
                     _receipt=receipt) == CCMD.EXIT_OK
    rec = json.loads(pathlib.Path(receipt).read_text())
    assert rec["report_path"] == COMBINE.COMBINED_REPORT
    assert rec["report_exists"] is False        # the stub wrote nothing
    assert len(rec["inputs"]) == R.N_SEGMENTS
    for k, entry in enumerate(rec["inputs"]):
        assert entry["segment"] == k
        assert entry["seed_block"] == list(RUN.SEGMENT_SEED_BLOCKS[k])
        for field in ("results_path", "trace_path", "report_path",
                      "receipt_path"):
            assert entry[field], field
    assert rec["n_games"] == 592 and rec["n_pairs"] == 296


def test_THE_RECEIPT_IS_CREATE_ONLY(tmp_path, fake_source, monkeypatch):
    """🔴 THE TWO RUNS MUST DIFFER, OR AN OVERWRITE LOOKS LIKE A NO-OP. Both
    runs used to return the same stub, so the receipt was byte-identical either
    way and the control removing the create-only guard was NOT CAUGHT -- the
    same trap as comparing an existing receipt before and after a refusal.

    The first run COMPLETES and the second FAILS, so an overwrite would replace
    a COMPLETED record with a FAILED one and cannot hide."""
    calls = []

    def once():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("the second attempt exploded")
        return {"n_games": 592, "n_pairs": 296, "verdict": "NO VERDICT"}

    monkeypatch.setattr(COMBINE, "combine", once)
    receipt = str(tmp_path / "out" / "receipt.json")
    assert CCMD.main(["--run"], _combine_source=fake_source,
                     _receipt=receipt) == CCMD.EXIT_OK
    first = json.loads(pathlib.Path(receipt).read_text())
    assert first["outcome"] == "COMPLETED" and first["n_games"] == 592

    pathlib.Path(fake_source).write_text("# header\n" + GATE_OPEN_SRC)
    CCMD.main(["--run"], _combine_source=fake_source, _receipt=receipt)
    assert len(calls) == 2, "the second run must really have been attempted"

    after = json.loads(pathlib.Path(receipt).read_text())
    assert after == first, (
        "a second run must not overwrite the first receipt")
    assert after["outcome"] == "COMPLETED", (
        "the FAILED second attempt must not have replaced the COMPLETED record")


def test_THE_VERDICT_TRAVELS_WITH_ITS_POPULATION_LIMITATION():
    """🔴 THE SCOPE IS PART OF THE RESULT, NOT A FOOTNOTE TO IT. A stored
    verdict of INCUMBENT STRONGER that can be read without the population it
    holds over is the most quotable thing this study produced and the easiest
    to quote wrongly. The claim is persisted in the same file, and this asserts
    it is still there and still says what it says."""
    rep = json.loads(pathlib.Path(COMBINE.COMBINED_REPORT).read_text())
    assert rep["verdict"] == "INCUMBENT STRONGER"
    assert rep["is_strength_verdict"] is True
    claim = rep["population"]["claim"]
    for phrase in ("UNIFORMLY AT RANDOM",
                   "says NOTHING about realistic play",
                   "not a position anyone plays"):
        assert phrase in claim, phrase
    assert rep["population"]["stratum"] == R.STRATUM_UNIFORM
    assert "NOMINAL UNDER THE DECLARED PAIR-INDEPENDENCE MODEL" in \
        rep["primary"]["interval_note"]


def test_THE_COMBINER_PERSISTS_THE_CLAIM_IT_IS_GIVEN(study):
    """🔴 THE FILE AND THE CODE THAT WRITES IT ARE TWO CLAIMS. The test above
    reads the report that already exists, so it says nothing about the combiner
    -- a change that stopped persisting the population claim would leave that
    file untouched and go unnoticed. This exercises the WRITER, on synthetic
    inputs, and requires the whole claim through."""
    out = run(study)
    assert out["population"]["stratum"] == R.STRATUM_UNIFORM
    claim = out["population"]["claim"]
    for phrase in ("UNIFORMLY AT RANDOM",
                   "says NOTHING about realistic play",
                   "not a position anyone plays"):
        assert phrase in claim, phrase


def test_THE_PAYLOAD_AND_THE_DIRECTORY_ARE_BOTH_FSYNCED(study, monkeypatch):
    """🔴 DURABILITY CANNOT BE TESTED BY CRASHING THE MACHINE, so it is tested
    at the syscall: the payload must reach the device before anything claims the
    official name, and the DIRECTORY entry must reach it too or the new name can
    vanish in a crash while the file itself survives.

    Spying on `os.fsync` is behavioural -- it observes what the code does, not
    what its source says."""
    synced = []
    real_fsync = COMBINE.os.fsync
    real_open = COMBINE.os.open
    fds = {}

    def spy_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        fds[fd] = path
        return fd

    def spy_fsync(fd):
        synced.append(fds.get(fd, "<unknown>"))
        return real_fsync(fd)

    monkeypatch.setattr(COMBINE.os, "open", spy_open)
    monkeypatch.setattr(COMBINE.os, "fsync", spy_fsync)
    run(study)

    out = study["out"]
    assert any(p.startswith(os.path.join(out, ".")) for p in synced), (
        "the payload file was never fsynced")
    assert out in synced or out.rstrip("/") in synced, (
        "the destination directory was never fsynced")


def test_THE_LINK_IS_A_BARRIER_IN_ITS_OWN_RIGHT(study, monkeypatch):
    """🔴 TWO GUARDS REFUSE AN OCCUPIED DESTINATION, AND ONE HID THE OTHER.
    `_occupied` checks first and `os.link` refuses atomically; while they were
    fused, a control replacing the install with `os.replace` -- which silently
    OVERWRITES -- passed, because the check had already returned. With the
    check suppressed, only the install can refuse, which is also the real
    race: a file appearing between the check and the install."""
    run(study)
    monkeypatch.setattr(COMBINE, "_occupied", lambda path: False)
    with pytest.raises(COMBINE.H3CombineError, match="CREATED WHILE THIS"):
        run(study)
    final = pathlib.Path(study["out"], os.path.basename(COMBINE.COMBINED_REPORT))
    assert json.loads(final.read_text())["n_games"] == R.N_GAMES, (
        "the first report must still be there, unmodified")


# ══════════ the PUBLIC entry, with its gate OPEN, on safe outputs ═══════════
def _open_gate_state(monkeypatch, study, tmp_path, gates):
    """Put the process in the state a REAL combination is in: the gate open, the
    inventory reporting it open, and the destination redirected somewhere safe.

    🔴 THE INPUTS ARE THE SYNTHETIC ONES. `combine()` resolves its inputs
    through `CMD.default_paths`, so those are redirected too -- pooling the REAL
    592 games here would compute the study's actual estimate as a side effect of
    a test, which is the one act the gate exists to hold.
    """
    monkeypatch.setattr(COMBINE, "H3_COMBINATION_AUTHORIZED", True)
    monkeypatch.setattr(COMBINE, "COMBINED_OUT_DIR", str(tmp_path / "safe"))
    monkeypatch.setattr(COMBINE.CMD, "default_paths", study["paths_for"])
    monkeypatch.setattr(COMBINE.CMD, "receipt_path", study["receipt_for"])
    from scripts.GPU.alphazero import gate_inventory as INV
    monkeypatch.setattr(INV, "open_gates", lambda: list(gates))


def test_THE_PUBLIC_ENTRY_ACCEPTS_ITS_OWN_OPEN_GATE(study, tmp_path, monkeypatch):
    """🔴 THE STATE NO TEST HAD EVER PUT IT IN. Every other test drives
    `combine_unguarded`, which needs no gate, so all eleven gates are shut and
    the final-state check trivially passes. The real invocation must have THIS
    gate open -- and the check rejected any open gate, so the combination
    refused itself on 2026-09-20 having never been exercised in the only state
    it can actually run in."""
    #: 🔴 THE OPEN GATE IS NAMED LITERALLY, NOT TAKEN FROM `THIS_GATE`. Using
    #: the constant under test to build the world the test runs in makes the
    #: test agree with itself: emptying THIS_GATE also emptied the simulated
    #: inventory, so no gate was open, nothing needed permitting, and the
    #: control that blanked the allow-list was NOT CAUGHT. The expectation may
    #: never come from the thing under test.
    _open_gate_state(monkeypatch, study, tmp_path,
                     [("h3_combine", "H3_COMBINATION_AUTHORIZED")])
    out = COMBINE.combine()
    assert out["n_games"] == R.N_GAMES == 592
    assert out["n_pairs"] == R.N_PAIRS == 296
    written = pathlib.Path(tmp_path / "safe"
                           / os.path.basename(COMBINE.COMBINED_REPORT))
    assert json.loads(written.read_text()) == out


def test_THE_PUBLIC_ENTRY_STILL_REFUSES_ANY_OTHER_OPEN_GATE(study, tmp_path,
                                                            monkeypatch):
    """The allow-list permits exactly one gate. An execution gate left open
    while the study is combined is still a disagreement."""
    _open_gate_state(monkeypatch, study, tmp_path,
                     [("h3_combine", "H3_COMBINATION_AUTHORIZED"),
                      ("h3_study_runner", "H3_STUDY_EXECUTION_AUTHORIZED")])
    with pytest.raises(COMBINE.H3CombineError, match="gates are OPEN"):
        COMBINE.combine()
    assert not os.path.lexists(tmp_path / "safe"), "a refusal writes nothing"


def test_THE_PUBLIC_ENTRY_STILL_REQUIRES_ITS_GATE_BEFORE_THE_BODY(study, tmp_path,
                                                                  monkeypatch):
    """The allow-list must not have turned the gate itself into a formality."""
    _open_gate_state(monkeypatch, study, tmp_path,
                     [("h3_combine", "H3_COMBINATION_AUTHORIZED")])
    monkeypatch.setattr(COMBINE, "H3_COMBINATION_AUTHORIZED", False)
    with pytest.raises(COMBINE.H3CombineError, match="NOT AUTHORIZED"):
        COMBINE.combine()
    assert not os.path.lexists(tmp_path / "safe")


def test_RESTORATION_RETURNS_THE_SOURCE_BYTE_FOR_BYTE(tmp_path):
    """🔴 IT DELETED A BLANK LINE. `\\s*$` under re.MULTILINE consumes the
    newline and any blank lines after it, so restoring the gate on 2026-09-20
    also removed one from h3_combine.py. The whole file must come back
    unchanged except for the one intended True -> False."""
    real = pathlib.Path(CCMD.COMBINE_SOURCE).read_bytes()
    assert b"\nH3_COMBINATION_AUTHORIZED = False\n" in real

    opened = real.replace(b"\nH3_COMBINATION_AUTHORIZED = False\n",
                          b"\nH3_COMBINATION_AUTHORIZED = True\n")
    assert opened != real
    copy = tmp_path / "h3_combine_copy.py"
    copy.write_bytes(opened)

    assert CCMD.restore_gate(str(copy)) is True
    assert CCMD.gate_readback(str(copy)) == "False"
    assert copy.read_bytes() == real, (
        "the restored file must differ from the original in NOTHING")


def test_ATTEMPT_1S_DESTINATION_IS_SPENT_AND_ATTEMPT_2S_IS_FRESH():
    """Attempt 1 was consumed by ATTEMPT, not by success: it wrote no report and
    its create-only receipt stands there as the record."""
    assert COMBINE.SPENT_COMBINED_DIRS
    for spent in COMBINE.SPENT_COMBINED_DIRS:
        assert os.path.lexists(spent), "the spent attempt's record must survive"
        assert os.path.lexists(os.path.join(spent, "00_combination_receipt.json"))
        assert not os.path.lexists(os.path.join(spent, "09_combined_report.json"))
        assert COMBINE.COMBINED_OUT_DIR != spent
        assert not COMBINE.COMBINED_OUT_DIR.startswith(spent.rstrip("/") + "/")
    #: attempt 2 has since RUN, so its destination holds the result rather than
    #: being absent. What still holds is that it is a DIFFERENT place.
    assert os.path.lexists(COMBINE.COMBINED_REPORT)
    assert os.path.lexists(CCMD.RECEIPT)


def test_AN_OPEN_GATE_IS_CAUGHT_FINAL(study, monkeypatch):
    """The allow-list narrowed the rule; it must not have removed it. An
    unexpected open gate is still a disagreement, with the default empty
    allow-list and with a non-empty one."""
    from scripts.GPU.alphazero import gate_inventory as INV
    from scripts.GPU.alphazero import h3_final_state as FINAL
    monkeypatch.setattr(INV, "open_gates",
                        lambda: [("h3_study_runner", "H3_STUDY_EXECUTION_AUTHORIZED")])
    for allow in (None, COMBINE.THIS_GATE):
        problems = FINAL.verify_final_state(
            paths_for=study["paths_for"], receipt_for=study["receipt_for"],
            allow_open=allow)
        assert any("gates are OPEN" in p for p in problems), (
            "an unexpected open gate must still be reported")


def test_THE_COMBINED_OUTPUTS_ARE_REHASHED_NOT_READ_BACK(tmp_path, monkeypatch):
    """🔴 A DIGEST THAT IS NEVER RECOMPUTED FROM THE THING IT DIGESTS IS A LABEL.
    Both directions: a file whose bytes no longer match its pin must be
    reported, and a pinned file that has gone missing must be reported too."""
    from scripts.GPU.alphazero import h3_final_state as FINAL
    assert FINAL.verify_final_state() == []

    tampered = tmp_path / "tampered.json"
    tampered.write_text(pathlib.Path(COMBINE.COMBINED_REPORT).read_text() + " ")
    monkeypatch.setattr(COMBINE, "COMBINED_REPORT", str(tampered))
    problems = FINAL.verify_final_state()
    assert any("combined report hashes to" in p for p in problems), problems

    monkeypatch.setattr(COMBINE, "COMBINED_REPORT", str(tmp_path / "gone.json"))
    problems = FINAL.verify_final_state()
    assert any("pinned but MISSING" in p for p in problems), problems
