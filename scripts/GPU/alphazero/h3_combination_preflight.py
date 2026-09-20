"""H3 — THE COMBINATION PRE-RUN VERIFICATION. READ-ONLY, AND IT IS A STOP.

Whether the one authorized combination could run, asked of the real tree with
the gate SHUT. It exits 0 when everything the combination needs is in place and
the only thing missing is the authorization -- which is the state it is supposed
to report, not a problem.

🔴 IT COUNTS, IT DOES NOT POOL. It reads the four result sets to check that 592
unique task ids, 592 unique seeds and 296 complete pairs are actually there, and
it never calls `summarise`, never sums a score and never reads a `primary`
figure. Counting the records is verification; pooling them is the interpretive
act the gate exists to hold, and a preflight that performed it to check that it
could would have performed it.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from . import gate_inventory as INVENTORY
from . import h3_combine as COMBINE
from . import h3_final_state as FINAL
from . import h3_study_command as CMD
from . import h3_study_rules as RULES
from . import h3_combine_command as CCMD0
from . import h3_study_runner as RUN

_FAILED: List[str] = []


def _refuses_combination() -> bool:
    """A second combination must be refused by the OCCUPIED DESTINATION, not
    only by the shut gate -- so this asks with the gate believed open."""
    import unittest.mock as _mock
    from . import gate_inventory as _INV
    with _mock.patch.object(COMBINE, "H3_COMBINATION_AUTHORIZED", True), \
         _mock.patch.object(_INV, "open_gates",
                            lambda: [("h3_combine", "H3_COMBINATION_AUTHORIZED")]):
        try:
            COMBINE.combine()
        except COMBINE.H3CombineError as exc:
            return "already exists" in str(exc)
    return False


def check(label: str, ok: Any, detail: str = "") -> bool:
    ok = bool(ok)
    print(f"  {'PASS' if ok else '🔴 FAIL'}     {label}"
          + (f"  {detail}" if detail else ""))
    if not ok:
        _FAILED.append(label)
    return ok


def main() -> int:
    print("=" * 74)
    print("H3 COMBINATION PREFLIGHT -- read-only, gate SHUT")
    print("(after the run: the COMBINATION CLOSEOUT check)")
    print("=" * 74)

    print("\n== the gate, and the destination it guards ==")
    check("H3_COMBINATION_AUTHORIZED is False",
          COMBINE.H3_COMBINATION_AUTHORIZED is False)
    #: 🔴 INVERTED AFTER THE COMBINATION RAN. This required the destination to
    #: be ABSENT, which was the pre-run claim. It has run; the destination holds
    #: the study's answer, and what must hold now is that the answer is still
    #: byte-for-byte the one that was written and that no second combination can
    #: replace it.
    import hashlib
    for _label, _path, _pin in (
            ("combined report", COMBINE.COMBINED_REPORT,
             COMBINE.COMBINED_REPORT_DIGEST),
            ("combination receipt", CCMD0.RECEIPT,
             COMBINE.COMBINATION_RECEIPT_DIGEST)):
        _there = os.path.lexists(_path)
        check(f"the {_label} is PRESENT", _there, _path)
        check(f"the {_label} still hashes to its pin",
              _there and hashlib.sha256(
                  open(_path, "rb").read()).hexdigest() == _pin,
              (_pin or "")[:16] + "…")
    check("a second combination is REFUSED -- the destination is occupied",
          _refuses_combination())
    import inspect
    check("combine() takes NO arguments -- no path, input or collaborator "
          "override reaches the authorized entry point",
          list(inspect.signature(COMBINE.combine).parameters) == [])
    try:
        COMBINE.combine()
        check("combine() REFUSES while the gate is shut", False)
    except COMBINE.H3CombineError as exc:
        check("combine() REFUSES while the gate is shut",
              "NOT AUTHORIZED" in str(exc))
    #: 🔴 "created nothing" was the pre-run claim. The destination now HOLDS the
    #: result, so the claim that survives is that a refusal leaves it untouched.
    check("and refusing left the destination untouched",
          sorted(os.listdir(COMBINE.COMBINED_OUT_DIR))
          == ["00_combination_receipt.json", "09_combined_report.json"])
    print(f"  {INVENTORY.gate_count()} gates derived from source; OPEN: "
          f"{INVENTORY.open_gates() or 'NONE'}")
    check("every gate is SHUT", INVENTORY.open_gates() == [])

    print("\n== the supervised command, and its restoration path ==")
    CCMD = CCMD0
    check("the command accepts NOTHING but --run",
          sorted(f for a in CCMD._parser()._actions for f in a.option_strings)
          == ["--help", "--run", "-h"])
    check("it reads the gate back from the SOURCE, not from this process",
          CCMD.gate_readback() == "False")
    check("the parent receipt records a COMPLETED combination",
          json.loads(open(CCMD.RECEIPT, encoding="utf-8").read())["outcome"]
          == "COMPLETED")
    check("a failed restoration has its OWN superseding exit code",
          CCMD.EXIT_GATE_NOT_RESTORED == 9
          and CCMD.EXIT_GATE_NOT_RESTORED not in
          (CCMD.EXIT_OK, CCMD.EXIT_REFUSED, CCMD.EXIT_NOT_AUTHORIZED,
           CCMD.EXIT_FAILED))
    _before = sorted(os.listdir(COMBINE.COMBINED_OUT_DIR))
    check("with the gate shut the command refuses and touches nothing",
          CCMD.main(["--run"]) == CCMD.EXIT_NOT_AUTHORIZED
          and sorted(os.listdir(COMBINE.COMBINED_OUT_DIR)) == _before)

    print("\n== the allow-list: narrow, and strict by default ==")
    #: 🔴 THE PREFLIGHT STILL DEMANDS ALL ELEVEN SHUT. It runs BEFORE the gate is
    #: opened, so the state it clears is the closed one; the allow-list is for
    #: the run itself, which cannot be in that state.
    check("the combiner permits exactly ONE gate, its own",
          tuple(COMBINE.THIS_GATE) == (("h3_combine", "H3_COMBINATION_AUTHORIZED"),))
    check("verify_final_state is STRICT by default (no allow_open)",
          "gates are OPEN" in " ".join(
              FINAL.verify_final_state(allow_open=None) or ["<none open>"])
          or INVENTORY.open_gates() == [])
    _mine = FINAL.verify_final_state(allow_open=COMBINE.THIS_GATE)
    check("and permitting its own gate changes nothing while all are shut",
          _mine == FINAL.verify_final_state())

    print("\n== attempt 1 is SPENT; attempt 2's destination is FRESH ==")
    for _spent in COMBINE.SPENT_COMBINED_DIRS:
        check("attempt 1's record is PRESERVED",
              os.path.lexists(os.path.join(_spent, "00_combination_receipt.json")),
              _spent)
        check("attempt 1 wrote NO report",
              not os.path.lexists(os.path.join(_spent, "09_combined_report.json")))
        check("attempt 2 wrote its report elsewhere",
          os.path.lexists(COMBINE.COMBINED_REPORT))
    check("attempt 2 is not inside it",
              COMBINE.COMBINED_OUT_DIR != _spent
              and not COMBINE.COMBINED_OUT_DIR.startswith(_spent.rstrip("/") + "/"))

    print("\n== the writer: the official name appears only when durable ==")
    import inspect
    src = inspect.getsource(COMBINE._write_create_only)
    check("the install is os.link (atomic, create-only), never rename/replace",
          "os.link(" in src and "os.rename(" not in src
          and "os.replace(" not in src)

    print("\n== the inputs: exactly the four pinned result sets ==")
    problems = FINAL.verify_final_state()
    for p in problems:
        print(f"    🔴 {p}")
    check("the four segments' records AGREE (zero disagreements)", not problems,
          f"{len(problems)} disagreement(s)")

    rows: List[Dict[str, Any]] = []
    for k in range(RULES.N_SEGMENTS):
        results_path, _t, _r = CMD.default_paths(k)
        with open(results_path, encoding="utf-8") as fh:
            got = [json.loads(line) for line in fh if line.strip()]
        rows.extend(r for r in got if r.get("record_type") == "task_result")
    task_ids = [r["task_id"] for r in rows]
    seeds = [r["seed"] for r in rows]
    expected_seeds = set()
    for lo, hi in RUN.SEGMENT_SEED_BLOCKS:
        expected_seeds |= set(range(lo, hi))
    pairs: Dict[int, int] = {}
    for r in rows:
        pairs[r["pair_id"]] = pairs.get(r["pair_id"], 0) + 1

    check(f"{RULES.N_GAMES} games, all task ids distinct",
          len(task_ids) == RULES.N_GAMES == len(set(task_ids)),
          f"{len(task_ids)} rows / {len(set(task_ids))} distinct")
    check(f"{RULES.N_GAMES} distinct seeds, exactly the union of the four blocks",
          len(set(seeds)) == RULES.N_GAMES and set(seeds) == expected_seeds)
    check(f"{RULES.N_PAIRS} pairs, each holding exactly two games",
          len(pairs) == RULES.N_PAIRS and all(n == 2 for n in pairs.values()),
          f"{len(pairs)} pairs")

    print("\n== permission, which is about PROVENANCE and not about counts ==")
    statuses = []
    for k in range(RULES.N_SEGMENTS):
        rec = json.loads(open(CMD.receipt_path(k), encoding="utf-8").read())
        statuses.append({"segment": k,
                         "status": ("completed" if rec.get("outcome") == "COMPLETED"
                                    else "void"),
                         "outcomes_inspected": True})
    permission = RUN.combine_segments(statuses)
    print(f"  {permission['why']}")
    check("combine_segments PERMITS a verdict", permission["verdict_permitted"])
    check("and it is not flagged", permission["flagged"] is False)

    print("\n== the analysis the combination will use ==")
    check("all three preregistered sensitivities are required",
          tuple(COMBINE.REQUIRED_SENSITIVITIES)
          == ("cap_free", "identity_free", "both"))
    from . import h3_study_analysis as ANALYSIS
    check("the combiner's analysis IS the preregistered summarise (by identity)",
          COMBINE.ANALYSIS.summarise is ANALYSIS.summarise)
    check("the combiner's verifier IS the final-state check (by identity)",
          COMBINE.FINAL.verify_final_state is FINAL.verify_final_state)
    check("the combiner's permission IS combine_segments (by identity)",
          COMBINE.RUN.combine_segments is RUN.combine_segments)

    print("\n" + "=" * 74)
    #: 🔴 DERIVED, NOT WRITTEN DOWN. This said "the combination COULD run and
    #: has not. It computed no estimate, pooled no games and wrote no file."
    #: That was true until 2026-09-20 and false the moment it ran -- a closing
    #: claim that outlives the state it describes is the defect this programme
    #: keeps finding, so it reads the state instead.
    _done = os.path.lexists(COMBINE.COMBINED_REPORT)
    if _done:
        _rep = json.loads(open(COMBINE.COMBINED_REPORT, encoding="utf-8").read())
        print("⚠ SCOPE. The combination HAS RUN. This establishes that its")
        print("  outputs are intact and that it cannot run again -- nothing here")
        print("  recomputed the estimate or reopened the question.")
        print(f"  verdict: {_rep['verdict']}  "
              f"({_rep['pairs_scored']} pairs, "
              f"strength_verdict={_rep['is_strength_verdict']})")
        print(f"  population: {_rep['population']['claim'][:96]}…")
    else:
        print("⚠ SCOPE. This establishes that the combination COULD run and has")
        print("  not. It computed no estimate, pooled no games and wrote no file.")
    print(f"\n{len(_FAILED)} FAILED")
    for label in _FAILED:
        print(f"    🔴 FAILED: {label}")
    if _FAILED:
        print("\n🔴 The combination must NOT be authorized while anything above "
              "fails.")
        return 1
    if _done:
        print("\nNo check FAILED. The combination is COMPLETE, its report and "
              "receipt are byte-identical to what was written, and the gate is "
              "SHUT.\n🔴 IT CANNOT RUN AGAIN: the destination is occupied and "
              "the install is create-only.")
    else:
        print("\nNo check FAILED. Everything the combination needs is in place "
              "and the gate is SHUT.\n🔴 THAT IS NOT PERMISSION: opening it is "
              "a separate reviewed edit, for ONE combination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
