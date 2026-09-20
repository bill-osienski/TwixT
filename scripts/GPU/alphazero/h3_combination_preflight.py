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
from . import h3_study_runner as RUN

_FAILED: List[str] = []


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
    print("=" * 74)

    print("\n== the gate, and the destination it guards ==")
    check("H3_COMBINATION_AUTHORIZED is False",
          COMBINE.H3_COMBINATION_AUTHORIZED is False)
    check("the combined destination is ABSENT",
          not os.path.lexists(COMBINE.COMBINED_OUT_DIR)
          and not os.path.lexists(COMBINE.COMBINED_REPORT),
          COMBINE.COMBINED_REPORT)
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
    check("and refusing created nothing",
          not os.path.lexists(COMBINE.COMBINED_OUT_DIR))
    print(f"  {INVENTORY.gate_count()} gates derived from source; OPEN: "
          f"{INVENTORY.open_gates() or 'NONE'}")
    check("every gate is SHUT", INVENTORY.open_gates() == [])

    print("\n== the supervised command, and its restoration path ==")
    from . import h3_combine_command as CCMD
    check("the command accepts NOTHING but --run",
          sorted(f for a in CCMD._parser()._actions for f in a.option_strings)
          == ["--help", "--run", "-h"])
    check("it reads the gate back from the SOURCE, not from this process",
          CCMD.gate_readback() == "False")
    check("the parent receipt is ABSENT", not os.path.lexists(CCMD.RECEIPT),
          CCMD.RECEIPT)
    check("a failed restoration has its OWN superseding exit code",
          CCMD.EXIT_GATE_NOT_RESTORED == 9
          and CCMD.EXIT_GATE_NOT_RESTORED not in
          (CCMD.EXIT_OK, CCMD.EXIT_REFUSED, CCMD.EXIT_NOT_AUTHORIZED,
           CCMD.EXIT_FAILED))
    check("with the gate shut the command refuses and writes nothing",
          CCMD.main(["--run"]) == CCMD.EXIT_NOT_AUTHORIZED
          and not os.path.lexists(CCMD.RECEIPT)
          and not os.path.lexists(COMBINE.COMBINED_OUT_DIR))

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
    print("⚠ SCOPE. This establishes that the combination COULD run and has not.")
    print("  It computed no estimate, pooled no games and wrote no file.")
    print(f"\n{len(_FAILED)} FAILED")
    for label in _FAILED:
        print(f"    🔴 FAILED: {label}")
    if _FAILED:
        print("\n🔴 The combination must NOT be authorized while anything above "
              "fails.")
        return 1
    print("\nNo check FAILED. Everything the combination needs is in place and "
          "the gate is SHUT.\n🔴 THAT IS NOT PERMISSION: opening it is a "
          "separate reviewed edit, for ONE combination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
