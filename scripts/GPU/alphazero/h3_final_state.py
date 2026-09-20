"""H3 FULL STUDY — THE FINAL-STATE VERIFICATION. Read-only, and it is a STOP.

🔴 WHY A SEPARATE CHECK, WHEN EVERY SEGMENT ALREADY REPORTED ITSELF.

Each segment's wrapper verified its own run and wrote its own receipt. Nothing
verified that the FOUR RECORDS AGREE WITH EACH OTHER AND WITH THE REGISTRY.
They are four independent runs, days apart, across edits to the runner, the
tests and the controls; "each one said OK" is not the same claim as "together
they describe one 592-game study that was actually played as designed".

WHAT IT CROSS-CHECKS, and every one of these is a way the set could disagree
while each part looks fine on its own:

  receipts    every segment COMPLETED, exit 0, worker exit 0, gate restored AND
              read back False, process group cleared, not interrupted, not
              timed out -- and the receipt's OWN `segment` field matches the
              directory it sits in.
  traces      run_start + N task_start + N task_done + run_end, terminal
              verdict OK, games_completed == N, and the trace's own segment.
  results     1 header + N transcripts + N task_results, and the N seeds
              CARRIED BY THE RECORDS are exactly the segment's block -- read
              off the rows, never re-derived from the plan.
  reports     N games, N/2 pairs, floor not reached, interpretation withheld,
              no strength verdict, complete, not timed out.
  pins        each segment's task digest recomputes to its pinned value, and
              the full schedule to `SCHEDULE_DIGEST`; the frozen population
              still hashes to `OPENING_SET_DIGEST`.
  seeds       every block EXPOSED whole and RETIRED whole, the four blocks
              disjoint, and their union exactly N_GAMES distinct seeds.
  gates       every authorization constant in the derived inventory is False.

🔑 IT RETURNS PROBLEMS, IT DOES NOT RAISE ON THE FIRST ONE. A verification that
stops at the first disagreement tells you one thing and hides the rest, and the
whole point here is to see whether the set is coherent.

🔴 IT COMPUTES NOTHING ABOUT STRENGTH. It never pools the segments, never sums a
score and never reads a `primary` figure. Whether the four segments may be
combined and interpreted is a separate, later, separately authorized question;
this module exists to establish that they are a well-formed set BEFORE that
question is asked, not to answer it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import e4_screen_reference as REF
from . import gate_inventory as INVENTORY
from . import h3_study_command as CMD
from . import h3_study_generator as GEN
from . import h3_study_rules as RULES
from . import h3_study_runner as RUN


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_receipt(k: int, rec: Mapping[str, Any], out: List[str]) -> None:
    #: 🔴 `gate_readback` IS A TRI-STATE STRING, NOT A BOOLEAN: "True", "False",
    #: or None when the source could not be read at all. Expecting a boolean
    #: here reported all four receipts as wrong when they were right -- and the
    #: string matters, because `if rec["gate_readback"]:` is TRUE for "False".
    #: Nothing in the tree does that today; this comment is why it must not start.
    want = {"outcome": "COMPLETED", "exit_code": 0, "worker_exit": 0,
            "gate_restored": True, "gate_readback": "False", "group_cleared": True,
            "interrupted": False, "timed_out": False, "segment": k,
            "design": "H3_FULL_STUDY_SEGMENT"}
    for key, value in want.items():
        got = rec.get(key, "<missing>")
        #: type-strict: `0` and `False` compare equal, and a receipt that
        #: recorded the wrong TYPE has recorded the wrong thing.
        if got != value or type(got) is not type(value):
            out.append(f"segment {k} receipt: {key} is {got!r}, expected {value!r}")
    pin = RUN.SEGMENT_DIGESTS[k]
    if rec.get("segment_digest") != pin:
        out.append(f"segment {k} receipt: segment_digest {rec.get('segment_digest')!r} "
                   f"is not the pin {pin!r}")


def _check_trace(k: int, rows: Sequence[Mapping[str, Any]], n: int,
                 out: List[str]) -> None:
    events = [r.get("event") for r in rows]
    for name, expect in (("run_start", 1), ("task_start", n),
                         ("task_done", n), ("run_end", 1)):
        got = events.count(name)
        if got != expect:
            out.append(f"segment {k} trace: {got} {name} events, expected {expect}")
    if not rows:
        out.append(f"segment {k} trace: empty")
        return
    last = rows[-1]
    if last.get("event") != "run_end":
        out.append(f"segment {k} trace: terminal event is {last.get('event')!r}")
    if last.get("verdict") != "OK":
        out.append(f"segment {k} trace: terminal verdict {last.get('verdict')!r}")
    if last.get("games_completed") != n:
        out.append(f"segment {k} trace: games_completed {last.get('games_completed')!r}")
    if last.get("segment") != k:
        out.append(f"segment {k} trace: terminal record names segment "
                   f"{last.get('segment')!r}")


def _check_results(k: int, rows: Sequence[Mapping[str, Any]], n: int,
                   block: Sequence[int], out: List[str]) -> None:
    kinds = [r.get("record_type") for r in rows]
    for name, expect in (("header", 1), ("transcript", n), ("task_result", n)):
        got = kinds.count(name)
        if got != expect:
            out.append(f"segment {k} results: {got} {name} rows, expected {expect}")
    #: 🔴 EXPOSURE IS READ OFF THE RECORDS. Re-deriving it from the plan would
    #: show only that the plan agrees with itself.
    seeds = sorted(r["seed"] for r in rows if "seed" in r)
    lo, hi = block
    if seeds != list(range(lo, hi)):
        out.append(f"segment {k} results: the {len(seeds)} seeds carried by the "
                   f"records are not exactly [{lo}, {hi})")
    if len(set(seeds)) != len(seeds):
        out.append(f"segment {k} results: a seed is repeated")


def _check_report(k: int, rep: Mapping[str, Any], n: int, out: List[str]) -> None:
    want = {"segment": k, "games_seen": n, "games_scored": n,
            "pairs_scored": n // 2, "below_report_floor": True,
            "interpretation_withheld": True, "is_strength_verdict": False,
            "verdict": "NO VERDICT", "complete": True, "timed_out": False,
            "pairs_excluded_incomplete": 0, "pairs_excluded_void": 0}
    for key, value in want.items():
        got = rep.get(key, "<missing>")
        if got != value or type(got) is not type(value):
            out.append(f"segment {k} report: {key} is {got!r}, expected {value!r}")
    for gate in rep.get("gates", []):
        if gate.get("status") != "CLEAR":
            out.append(f"segment {k} report: gate {gate.get('rule')!r} is "
                       f"{gate.get('status')!r}")


def verify_final_state(*, paths_for=None, receipt_for=None,
                       blocks: Optional[Sequence] = None) -> List[str]:
    """Every disagreement across the four segments' records. `[]` means they
    agree. The three injection points exist so a test can feed a TAMPERED record
    and watch a specific cross-check fail, without touching real evidence.
    """
    paths_for = paths_for or CMD.default_paths
    receipt_for = receipt_for or CMD.receipt_path
    blocks = tuple(blocks if blocks is not None else RUN.SEGMENT_SEED_BLOCKS)
    n = RULES.GAMES_PER_SEGMENT
    out: List[str] = []

    if len(blocks) != RULES.N_SEGMENTS:
        out.append(f"{len(blocks)} seed blocks, expected {RULES.N_SEGMENTS}")
        return out

    for k, block in enumerate(blocks):
        results_path, trace_path, report_path = paths_for(k)
        receipt_path = receipt_for(k)
        for label, path in (("receipt", receipt_path), ("results", results_path),
                            ("trace", trace_path), ("report", report_path)):
            if not os.path.lexists(path):
                out.append(f"segment {k}: {label} is missing at {path}")
        if any(f"segment {k}:" in m for m in out):
            continue
        _check_receipt(k, _load_json(receipt_path), out)
        _check_trace(k, _load_jsonl(trace_path), n, out)
        _check_results(k, _load_jsonl(results_path), n, block, out)
        _check_report(k, _load_json(report_path), n, out)

        #: the block itself: EXPOSED whole and RETIRED whole, the one-shot rule.
        lo, hi = block
        st = [REF.seed_status(x) for x in range(lo, hi)]
        if hi - lo != n:
            out.append(f"segment {k}: block is {hi - lo} wide, expected {n}")
        if not all(x["accounted"] for x in st):
            out.append(f"segment {k}: block is not wholly ACCOUNTED")
        if not all(x["exposed"] for x in st):
            out.append(f"segment {k}: block is not wholly EXPOSED")
        if not all(x["retired"] for x in st):
            out.append(f"segment {k}: block is not wholly RETIRED")

    #: the blocks together
    seen: set = set()
    for k, (lo, hi) in enumerate(blocks):
        rng = set(range(lo, hi))
        if rng & seen:
            out.append(f"segment {k}'s block overlaps an earlier one")
        seen |= rng
    if len(seen) != RULES.N_GAMES:
        out.append(f"the four blocks hold {len(seen)} seeds, expected {RULES.N_GAMES}")

    #: the pins, RECOMPUTED from the population rather than read back
    try:
        openings = RUN.load_opening_set(GEN.DEFAULT_OUT)
        if RULES.opening_set_digest(openings) != RULES.OPENING_SET_DIGEST:
            out.append("the frozen population no longer hashes to OPENING_SET_DIGEST")
        tasks = RULES.build_tasks(openings, seed_blocks=blocks)
        try:
            RUN.check_schedule_digest(tasks)
        except Exception as exc:                                 # noqa: BLE001
            out.append(f"the full schedule no longer matches SCHEDULE_DIGEST: {exc}")
        for k in range(RULES.N_SEGMENTS):
            if RUN.segment_digest(tasks, k) != RUN.SEGMENT_DIGESTS[k]:
                out.append(f"segment {k}'s task digest no longer recomputes to its pin")
    except Exception as exc:                                     # noqa: BLE001
        out.append(f"the pins could not be recomputed: {exc}")

    #: the gates, DERIVED from source
    open_gates = INVENTORY.open_gates()
    if open_gates:
        out.append(f"gates are OPEN: {open_gates}")
    #: 🔴 NO COUNT PINNED HERE, DELIBERATELY. It used to say `!= 10`. The gate
    #: TOTAL has exactly one tripwire -- `EXPECTED_GATES` in
    #: tests/test_gate_inventory.py -- and the reason it has one is that three
    #: hand-kept copies once claimed seven, eight and ten at the same time. What
    #: matters to the study's final state is that every gate is SHUT, which is
    #: derived above and cannot go stale when a gate is legitimately added.
    return out
