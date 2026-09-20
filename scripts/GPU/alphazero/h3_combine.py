"""H3 FULL STUDY — THE COMBINATION. GATE-SHUT, CREATE-ONLY, ONE SHOT.

This is the only place the four segments are ever pooled, and it exists because
pooling them is the single irreversible interpretive act of the study: after it,
a number exists that people will quote.

🔴 WHAT IT REFUSES, AND WHY EACH REFUSAL IS NOT A FORMALITY

  the gate            `H3_COMBINATION_AUTHORIZED` is False. `combine()` takes NO
                      ARGUMENTS -- no path override, no input override, no way
                      to aim it somewhere else. An entry point that accepts its
                      own seams is not a gate, so the seams live on
                      `combine_unguarded`, which the gate does not protect and
                      which the frozen entry point never exposes.
  the final state     `verify_final_state()` must report ZERO disagreements
                      first. Pooling records that do not agree with each other
                      produces a number that describes no run.
  the inputs          EXACTLY the four pinned result sets, read from the frozen
                      destinations, hashed, and recorded by path AND sha256.
  the counts          592 unique task ids, 592 unique seeds equal to the union
                      of the four blocks, and 296 pairs holding exactly two
                      games each. A pair short one game is not a pair.
  the permission      `combine_segments()` must PERMIT a verdict. It gates on
                      PROVENANCE -- optional stopping, VOIDs -- not on counts,
                      and no arithmetic here can substitute for it.
  the destination     create-only, and DURABLE BEFORE IT IS NAMED. The payload
                      is written and fsynced to a temporary file in the same
                      directory, then installed with `os.link()` -- atomic, and
                      refusing an existing name. After any interruption the
                      official path either does not exist or holds the whole
                      report; an exception handler cannot promise that, because
                      a SIGKILL does not run one.

🔑 `summarise()` IS CALLED EXACTLY ONCE, on the pooled games, and it is the
SAME preregistered function each segment used. Nothing here re-implements a
mean, a floor, a gate or an interval; a combiner with its own copy of the
analysis would be a second design competing with the frozen one.

⚠ THE REPORT THIS WRITES MAY STILL WITHHOLD. Reaching 296 pairs clears the
floor; it does not make a verdict appear. `summarise()` decides that, exactly as
it did for each quarter, and its answer is persisted whatever it is.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import e4_screen_reference as REF
from . import h3_final_state as FINAL
from . import h3_study_analysis as ANALYSIS
from . import h3_study_command as CMD
from . import h3_study_rules as RULES
from . import h3_study_runner as RUN

#: 🔴 THE ELEVENTH GATE. One reviewed edit opens it, for one combination.
H3_COMBINATION_AUTHORIZED = False
#: THE FIXED DESTINATION. Not a parameter of `combine()`, so a run cannot be
#: aimed anywhere else, and not reused: create-only means a second combination
#: needs a new reviewed destination and a new authorization.
COMBINED_OUT_DIR = f"{RUN.OUT_ROOT}/2026-09-20-t1j-h3-study-combined"
COMBINED_REPORT = f"{COMBINED_OUT_DIR}/09_combined_report.json"

#: the three preregistered sensitivities, by name. Persisting two of them and
#: calling it the report would be a quiet narrowing of the design.
REQUIRED_SENSITIVITIES = ("cap_free", "identity_free", "both")


class H3CombineError(RuntimeError):
    """A refusal from the combination. Never a report."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_results(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "task_result":
                out.append(row)
    return out


def _occupied(path: str) -> bool:
    """Whether `path` is taken. `lexists`, so a DANGLING SYMLINK counts -- it
    would be followed by an ordinary open and silently write somewhere else.

    🔑 A NAMED FUNCTION SO THE TWO BARRIERS CAN BE TESTED APART. This check and
    the atomic `os.link` install both refuse an occupied destination, and while
    they were fused a control that broke the install was silently covered by the
    check: the defect was real and the test still passed.
    """
    return os.path.lexists(path)


def _write_create_only(path: str, payload: Mapping[str, Any]) -> None:
    """Install the report at `path` ONLY once the complete payload is durable.

    🔴 AN EXCEPTION HANDLER DOES NOT SURVIVE A KILL. The first version opened
    the FINAL path with `O_EXCL` and serialised into it, removing the partial
    file in an `except`. That covers a Python exception and nothing else: a
    SIGKILL, a crash or a power loss between the open and the last byte would
    leave a TRUNCATED REPORT at the frozen one-shot destination -- and the next
    attempt would be refused for a file that never held a result.

    So the official path is never opened for writing at all:

      1. serialise into a temporary file IN THE SAME DIRECTORY, so the install
         cannot cross a filesystem;
      2. flush and `fsync` it -- the bytes are on the device before anything
         claims the name;
      3. install with `os.link()`, which is ATOMIC and fails if the destination
         exists. `os.rename`/`os.replace` would silently overwrite, which is
         the opposite of create-only;
      4. `fsync` the DIRECTORY, so the new name survives a crash too;
      5. unlink the temporary name, in a `finally`, so a failure anywhere above
         leaves litter rather than a half-installed report.

    After any interruption the official path either does not exist or holds the
    whole payload. There is no third state.
    """
    if _occupied(path):
        raise H3CombineError(
            f"{path} already exists. The combined report is written ONCE; a "
            f"second combination needs a new reviewed destination.")
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            #: 🔴 A DIFFERENT SITUATION, AND IT SAYS SO. `_occupied` refuses a
            #: destination that was ALREADY taken; reaching here means the name
            #: appeared while this combination was running. Giving both the same
            #: words made the early check undetectable -- removing it changed no
            #: observable behaviour, so no control could bind it.
            raise H3CombineError(
                f"{path} was CREATED WHILE THIS COMBINATION WAS RUNNING. The "
                f"install is atomic and refused it; nothing was overwritten.")
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def combine_unguarded(
        *, out_dir: str,
        paths_for: Optional[Callable] = None,
        receipt_for: Optional[Callable] = None,
        blocks: Optional[Sequence] = None,
        verify: Optional[Callable] = None,
        summarise: Optional[Callable] = None,
        permit: Optional[Callable] = None) -> Dict[str, Any]:
    """Pool the four segments and write the combined report. NO GATE HERE.

    The seams exist so the whole path can be exercised end to end on synthetic
    inputs and a temporary destination. `combine()` exposes NONE of them.
    """
    paths_for = paths_for or CMD.default_paths
    receipt_for = receipt_for or CMD.receipt_path
    blocks = tuple(blocks if blocks is not None else RUN.SEGMENT_SEED_BLOCKS)
    verify = verify or FINAL.verify_final_state
    summarise = summarise or ANALYSIS.summarise
    permit = permit or RUN.combine_segments

    # ── 1. the four records must agree BEFORE anything is pooled
    problems = verify(paths_for=paths_for, receipt_for=receipt_for, blocks=blocks)
    if problems:
        raise H3CombineError(
            f"the final state reports {len(problems)} disagreement(s); the "
            f"segments may not be pooled until they describe one study:\n  "
            + "\n  ".join(problems))

    # ── 2. exactly the four pinned result sets, hashed as they are read
    inputs: List[Dict[str, Any]] = []
    games: List[Dict[str, Any]] = []
    total_elapsed = 0.0
    for k, block in enumerate(blocks):
        results_path, trace_path, report_path = paths_for(k)
        receipt_path = receipt_for(k)
        rows = _task_results(results_path)
        report = json.loads(open(report_path, encoding="utf-8").read())
        total_elapsed += float(report["total_elapsed_s"])
        games.extend(rows)
        inputs.append({
            "segment": k,
            "seed_block": [block[0], block[1]],
            "games": len(rows),
            "results_path": results_path, "results_sha256": _sha256(results_path),
            "trace_path": trace_path, "trace_sha256": _sha256(trace_path),
            "report_path": report_path, "report_sha256": _sha256(report_path),
            "receipt_path": receipt_path, "receipt_sha256": _sha256(receipt_path),
        })

    # ── 3. the counts. 592 unique tasks, 592 unique seeds, 296 complete pairs
    task_ids = [g["task_id"] for g in games]
    if len(task_ids) != RULES.N_GAMES or len(set(task_ids)) != RULES.N_GAMES:
        raise H3CombineError(
            f"{len(task_ids)} games with {len(set(task_ids))} distinct task ids; "
            f"the study is {RULES.N_GAMES}. A missing or repeated game is a "
            f"harness fault, not a result.")
    seeds = [g["seed"] for g in games]
    expected_seeds = set()
    for lo, hi in blocks:
        expected_seeds |= set(range(lo, hi))
    if len(set(seeds)) != RULES.N_GAMES or set(seeds) != expected_seeds:
        raise H3CombineError(
            f"the {len(seeds)} records carry {len(set(seeds))} distinct seeds, "
            f"which are not exactly the union of the four blocks")
    by_pair: Dict[int, int] = {}
    for g in games:
        by_pair[g["pair_id"]] = by_pair.get(g["pair_id"], 0) + 1
    if len(by_pair) != RULES.N_PAIRS:
        raise H3CombineError(
            f"{len(by_pair)} pairs, expected {RULES.N_PAIRS}")
    short = sorted(p for p, n in by_pair.items() if n != 2)
    if short:
        raise H3CombineError(
            f"{len(short)} pair(s) do not hold exactly two games "
            f"(first: {short[:5]}); a half pair is not a pair")

    # ── 4. PERMISSION, which is about provenance and not about counts
    statuses = []
    for k in range(len(blocks)):
        rec = json.loads(open(receipt_for(k), encoding="utf-8").read())
        statuses.append({
            "segment": k,
            "status": "completed" if rec.get("outcome") == "COMPLETED" else "void",
            #: every segment's outcomes HAVE been inspected -- each one reported
            #: its own descriptive figures as it finished. Claiming otherwise
            #: here would be the study telling itself a convenient story.
            "outcomes_inspected": True,
        })
    permission = permit(statuses)
    if not permission.get("verdict_permitted"):
        raise H3CombineError(
            f"combine_segments REFUSES a verdict: {permission.get('why')}")

    # ── 5. the preregistered analysis, ONCE, on the pooled games
    report = summarise(games, total_elapsed_s=total_elapsed)

    missing = [s for s in REQUIRED_SENSITIVITIES if s not in report["sensitivities"]]
    if missing:
        raise H3CombineError(
            f"the report omits the preregistered sensitivit(y/ies) {missing}; "
            f"persisting a narrower analysis than the one frozen is not an "
            f"option the design leaves open")

    payload = {
        "design": "H3_FULL_STUDY_COMBINED",
        "n_segments": len(blocks),
        "n_games": len(games),
        "n_pairs": len(by_pair),
        "opening_set_digest": RULES.OPENING_SET_DIGEST,
        "schedule_digest": RUN.SCHEDULE_DIGEST,
        "segment_digests": list(RUN.SEGMENT_DIGESTS),
        "inputs": inputs,
        "permission": permission,
        "primary": report["primary"],
        "sensitivities": report["sensitivities"],
        "gates": report["gates"],
        "population": report["population"],
        "games_seen": report["games_seen"],
        "games_scored": report["games_scored"],
        "pairs_scored": report["pairs_scored"],
        "pairs_excluded_incomplete": report["pairs_excluded_incomplete"],
        "pairs_excluded_void": report["pairs_excluded_void"],
        "duplicate_pairs": report["duplicate_pairs"],
        "within_pair_identical": report["within_pair_identical"],
        "capped_games": report["capped_games"],
        "pairs_with_a_cap": report["pairs_with_a_cap"],
        "shared_continuation_pairs": report["shared_continuation_pairs"],
        "shared_continuation_relations": report["shared_continuation_relations"],
        "shared_continuation_note": report["shared_continuation_note"],
        "below_report_floor": report["below_report_floor"],
        "interpretation_withheld": report["interpretation_withheld"],
        "is_strength_verdict": report["is_strength_verdict"],
        "verdict": report["verdict"],
        "verdict_note": report["verdict_note"],
        "total_elapsed_s": report["total_elapsed_s"],
    }
    _write_create_only(os.path.join(out_dir, os.path.basename(COMBINED_REPORT)),
                       payload)
    return payload


def combine() -> Dict[str, Any]:
    """THE ONLY AUTHORIZED ENTRY POINT. It takes NO ARGUMENTS.

    🔴 THAT IS THE GATE'S SHAPE, NOT A CONVENIENCE. Every path, every input and
    every collaborator is frozen here. A caller cannot aim this at other
    records, a different destination, a looser verifier or a different analysis;
    if it could, opening the gate would authorize something other than the thing
    that was reviewed.
    """
    if not H3_COMBINATION_AUTHORIZED:
        raise H3CombineError(
            "the H3 COMBINATION is NOT AUTHORIZED (H3_COMBINATION_AUTHORIZED is "
            "False). Nothing was read, nothing was pooled and nothing was "
            "written. Combining is the study's one interpretive act and takes "
            "its own reviewed edit.")
    return combine_unguarded(out_dir=COMBINED_OUT_DIR)
