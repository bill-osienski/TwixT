"""H3 FULL STUDY — the FINAL-STATE cross-check.

Each segment's wrapper verified its own run. Nothing verified that the four
records agree with each other and with the registry, and "each one said OK" is
a weaker claim than "together these describe one 592-game study played as
designed".

Every tamper test below works on a COPY of the evidence in a temp directory and
reaches the real checker through its injection points. The real evidence is
never written to: these tests must be able to break things, and the thing they
would break is the only record of four runs that cannot be repeated.
"""
import copy
import json
import os
import pathlib
import shutil

import pytest

from scripts.GPU.alphazero import h3_final_state as F
from scripts.GPU.alphazero import h3_study_command as CMD
from scripts.GPU.alphazero import h3_study_runner as RUN
from scripts.GPU.alphazero import h3_study_rules as R


def test_THE_FOUR_SEGMENTS_AGREE():
    """The real thing, on the real evidence, read-only."""
    assert F.verify_final_state() == []


# ─────────────────────── a writable copy of the evidence ────────────────────
@pytest.fixture
def mirror(tmp_path):
    """The four segment directories, copied. Returns (paths_for, receipt_for)."""
    for k in range(R.N_SEGMENTS):
        dst = tmp_path / f"segment{k}"
        dst.mkdir()
        for src in list(CMD.default_paths(k)) + [CMD.receipt_path(k)]:
            shutil.copy2(src, dst / os.path.basename(src))

    def paths_for(k):
        return tuple(str(tmp_path / f"segment{k}" / os.path.basename(p))
                     for p in CMD.default_paths(k))

    def receipt_for(k):
        return str(tmp_path / f"segment{k}"
                   / os.path.basename(CMD.receipt_path(k)))

    return paths_for, receipt_for


def _problems(mirror):
    paths_for, receipt_for = mirror
    return F.verify_final_state(paths_for=paths_for, receipt_for=receipt_for)


def test_THE_MIRROR_ITSELF_AGREES(mirror):
    """🔴 THE CLEAN BASELINE. Every tamper test below is worthless if the copy
    is already failing -- it would 'catch' a defect it did not cause."""
    assert _problems(mirror) == []


def _edit_json(path, **fields):
    data = json.loads(pathlib.Path(path).read_text())
    data.update(fields)
    pathlib.Path(path).write_text(json.dumps(data))


def test_A_RECEIPT_THAT_DID_NOT_COMPLETE_IS_CAUGHT(mirror):
    paths_for, receipt_for = mirror
    _edit_json(receipt_for(2), outcome="TIMED_OUT")
    assert any("segment 2 receipt: outcome" in p for p in _problems(mirror))


def test_A_RECEIPT_WHOSE_GATE_WAS_LEFT_OPEN_IS_CAUGHT(mirror):
    """The wrapper restores the gate; a receipt saying otherwise is the whole
    reason the field exists."""
    paths_for, receipt_for = mirror
    _edit_json(receipt_for(0), gate_readback="True")
    assert any("segment 0 receipt: gate_readback" in p for p in _problems(mirror))


def test_A_RECEIPT_NAMING_THE_WRONG_SEGMENT_IS_CAUGHT(mirror):
    """A receipt sitting in segment 1's directory that calls itself segment 2
    means two runs wrote one record, or one run wrote the wrong one."""
    paths_for, receipt_for = mirror
    _edit_json(receipt_for(1), segment=2)
    assert any("segment 1 receipt: segment is 2" in p for p in _problems(mirror))


def test_A_RECEIPT_CARRYING_ANOTHER_SEGMENTS_DIGEST_IS_CAUGHT(mirror):
    paths_for, receipt_for = mirror
    _edit_json(receipt_for(3), segment_digest=RUN.SEGMENT_DIGESTS[0])
    assert any("segment 3 receipt: segment_digest" in p for p in _problems(mirror))


def test_A_MISSING_ARTIFACT_IS_CAUGHT(mirror):
    paths_for, receipt_for = mirror
    os.remove(paths_for(2)[1])                       # the trace
    assert any("segment 2: trace is missing" in p for p in _problems(mirror))


def test_A_TRACE_MISSING_ONE_GAME_IS_CAUGHT(mirror):
    """148 games or it is not the segment the pin describes."""
    paths_for, receipt_for = mirror
    trace = pathlib.Path(paths_for(0)[1])
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    dropped = [r for r in rows if r.get("event") != "task_done"][:1] + \
              [r for r in rows if r.get("event") == "task_done"][1:] + \
              [r for r in rows if r.get("event") in ("task_start",)]
    trace.write_text("\n".join(json.dumps(r) for r in dropped) + "\n")
    assert any("segment 0 trace:" in p for p in _problems(mirror))


def test_A_TRACE_THAT_DID_NOT_END_OK_IS_CAUGHT(mirror):
    paths_for, receipt_for = mirror
    trace = pathlib.Path(paths_for(1)[1])
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    rows[-1]["verdict"] = "VOID"
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert any("segment 1 trace: terminal verdict" in p for p in _problems(mirror))


def test_A_SEED_OUTSIDE_THE_BLOCK_IS_CAUGHT(mirror):
    """🔴 EXPOSURE IS READ OFF THE RECORDS. A record carrying a seed from
    somewhere else means the games played were not the games scheduled."""
    paths_for, receipt_for = mirror
    results = pathlib.Path(paths_for(2)[0])
    rows = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
    for r in rows:
        if "seed" in r:
            r["seed"] = 999_000_000
            break
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert any("segment 2 results: the 148 seeds" in p for p in _problems(mirror))


def test_A_REPORT_CLAIMING_A_STRENGTH_VERDICT_IS_CAUGHT(mirror):
    """Each segment is a quarter of the study; none of them may claim one."""
    paths_for, receipt_for = mirror
    _edit_json(paths_for(3)[2], is_strength_verdict=True,
               interpretation_withheld=False, verdict="INCUMBENT_STRONGER")
    problems = _problems(mirror)
    assert any("segment 3 report: is_strength_verdict" in p for p in problems)
    assert any("segment 3 report: interpretation_withheld" in p for p in problems)


def test_A_REPORT_WITH_A_FIRED_DEGENERACY_GATE_IS_CAUGHT(mirror):
    paths_for, receipt_for = mirror
    data = json.loads(pathlib.Path(paths_for(0)[2]).read_text())
    data["gates"][0]["status"] = "FIRED"
    pathlib.Path(paths_for(0)[2]).write_text(json.dumps(data))
    assert any("segment 0 report: gate" in p for p in _problems(mirror))


def test_AN_OPEN_GATE_IS_CAUGHT(mirror, monkeypatch):
    from scripts.GPU.alphazero import gate_inventory as INV
    monkeypatch.setattr(INV, "open_gates",
                        lambda: [("h3_study_runner", "H3_STUDY_EXECUTION_AUTHORIZED")])
    assert any("gates are OPEN" in p for p in _problems(mirror))


def test_A_BLOCK_THAT_IS_NOT_RETIRED_IS_CAUGHT(mirror, monkeypatch):
    """The one-shot rule: a block that ran must be spent whole."""
    paths_for, receipt_for = mirror
    fresh = (202_608_188, 202_608_188 + R.GAMES_PER_SEGMENT)   # accounted, unspent
    blocks = (fresh,) + tuple(RUN.SEGMENT_SEED_BLOCKS[1:])
    problems = F.verify_final_state(paths_for=paths_for, receipt_for=receipt_for,
                                    blocks=blocks)
    assert any("segment 0: block is not wholly EXPOSED" in p for p in problems)
    assert any("segment 0: block is not wholly RETIRED" in p for p in problems)


def test_IT_IGNORES_THE_STRENGTH_FIGURES_ENTIRELY(mirror):
    """🔴 BEHAVIOURAL, NOT A GREP. Rewrite every segment's primary figure to
    something absurd and the verdict of this check must not move: it establishes
    that the four records are a well-formed SET, and says nothing about who is
    stronger. Combining and interpreting is a separate, later question."""
    paths_for, receipt_for = mirror
    for k in range(R.N_SEGMENTS):
        path = pathlib.Path(paths_for(k)[2])
        data = json.loads(path.read_text())
        data["primary"] = dict(data["primary"], mean=0.0, n=1,
                               interval=[-9.0, 9.0], favours="t1j")
        for pair in data.get("pairs", []):
            pair["score"] = 0.0
        path.write_text(json.dumps(data))
    assert _problems(mirror) == [], (
        "the final-state check must not depend on the scores")
