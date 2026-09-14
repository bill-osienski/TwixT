#!/usr/bin/env python3
"""H3 PILOT — THE PRE-RUN VERIFICATION. READ-ONLY, and it is a STOP.

    .venv/bin/python -m scripts.GPU.alphazero.h3_prerun_verification

Measures the state a pilot-execution authorization would be given against, and
exits NONZERO if any check fails. A verification that prints FAIL and returns 0 is
the defect this programme spent 2026-09-12 removing from its own harness.

🔴 THE GATE IS SHUT AND MUST STAY SHUT WHILE THIS RUNS. No gate is opened, no seed
is drawn, no JVM starts, no model is loaded, no game is played and nothing is
written. The one effectful-looking step is `build_reference_agent` on the REAL
registered tasks with a STUB evaluator -- the call that aborted on H2's attempt 1
-- which constructs our own agent object and touches neither T1j nor a seed's RNG
stream. `restore_gate` is exercised on the real runner source, where the gate is
ALREADY closed, so it rewrites nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from . import e4_screen_reference as REF
from . import h3_pilot_analysis as ANALYSIS
from . import h3_pilot_command as CMD
from . import h3_pilot_rules as RULES
from . import h3_pilot_runner as RUN

#: The commit whose work this verifies. HEAD must DESCEND from it with a clean
#: SOURCE tree -- asserting equality would be circular, since this file and
#: anything it finds are commits on top.
BOUND_COMMIT = "f87e5d1"

GATES: Tuple[Tuple[str, str], ...] = (
    ("d1_probe", "D1_EXECUTION_AUTHORIZED"),
    ("e4_screen_command", "SCREEN_AUTHORIZED"),
    ("h1_viability_runner", "H1_EXECUTION_AUTHORIZED"),
    ("l0_match_command", "L0_EXECUTION_AUTHORIZED"),
    ("lowply_qualification", "LOWPLY_QUALIFICATION_AUTHORIZED"),
    ("runtime_requalification", "RUNTIME_REQUAL_AUTHORIZED"),
    ("h2_match_runner", "H2_EXECUTION_AUTHORIZED"),
    ("h3_pilot_runner", "H3_PILOT_EXECUTION_AUTHORIZED"),
)

#: The wrapper and runner tests that bind the claims below. RUN here, not cited:
#: a named test that does not pass proves nothing.
BINDING_TESTS = (
    "tests/test_h3_pilot_runner.py",
    "tests/test_h3_pilot_openings.py",
    "tests/test_h3_pilot_analysis.py",
)

_checks: List[Tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    _checks.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '🔴 FAIL':8s} {label}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    print(f"H3 PILOT PRE-RUN VERIFICATION -- bound to {BOUND_COMMIT}")
    print("Nothing has been executed. The gate is shut at this moment.")
    print("=" * 74)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines()
    ignored = [l for l in porcelain if l[3:].strip().startswith(CMD.OUT_DIR)]
    dirty = "\n".join(l for l in porcelain if l not in ignored).strip()
    descends = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BOUND_COMMIT, "HEAD"]).returncode == 0

    print("\n== the tree ==")
    print(f"  HEAD {head}, descends from {BOUND_COMMIT}")
    check(f"HEAD descends from the bound commit ({BOUND_COMMIT})", descends)
    if ignored:
        print(f"  ignoring {len(ignored)} path(s) under {CMD.OUT_DIR} (this tool's output)")
    check("no SOURCE differs from the commit", not dirty,
          "" if not dirty else f"{[l[3:] for l in dirty.splitlines()][:3]}")

    print("\n== the EIGHT gates ==")
    import importlib
    for mod_name, attr in GATES:
        mod = importlib.import_module(f"scripts.GPU.alphazero.{mod_name}")
        check(f"{attr:34s} False", getattr(mod, attr) is False)

    print("\n== the registered seed block ==")
    lo, hi = RUN.PILOT_SEED_BLOCK
    states = [REF.seed_status(s) for s in range(lo, hi)]
    acc = sum(s["accounted"] for s in states)
    exp = sum(s["exposed"] for s in states)
    ret = sum(s["retired"] for s in states)
    tst = sum(s["test_only"] for s in states)
    print(f"  [{lo}, {hi}): accounted {acc} / exposed {exp} / retired {ret} / "
          f"test_only {tst}")
    check("the block is 40 seeds", hi - lo == RULES.N_GAMES == 40)
    check("every seed ACCOUNTED", acc == 40)
    check("NOT exposed, NOT retired, NOT test-only -- a reservation is not a draw",
          (exp, ret, tst) == (0, 0, 0))
    check("no seed is in CONSUMED_SEEDS",
          not any(s in REF.CONSUMED_SEEDS for s in range(lo, hi)))
    try:
        RUN.check_seed_registration()
        check("the registration barrier is SATISFIED", True)
    except Exception as e:                                    # noqa: BLE001
        check("the registration barrier is SATISFIED", False, str(e)[:90])
    from . import h2_match_rules as H2R
    from .d1_selection import SEED_INTERVAL as D1_BLOCK
    ours = set(range(lo, hi))
    spent = {"H2 a1": H2R.H2_ATTEMPT1_SEED_BLOCK, "H2 a2": H2R.H2_ATTEMPT2_SEED_BLOCK,
             "H2 a3": H2R.H2_SEED_BLOCK, "D1 §14": D1_BLOCK}
    check("DISJOINT from every spent block, recomputed here",
          not any(ours & set(range(*b)) for b in spent.values()))

    print("\n== the openings, and the schedule they fix ==")
    openings = RULES.generate_openings()
    check("20 distinct openings at the frozen depth",
          len(openings) == 20 and len({o["digest"] for o in openings}) == 20)
    check("the opening SET matches its pin",
          RULES.opening_set_digest(openings) == RULES.OPENING_SET_DIGEST)
    tasks = RULES.build_tasks(openings, seed_interval=RUN.PILOT_SEED_BLOCK)
    seeded_digest = RULES.task_digest(tasks)
    print(f"  seeded task digest : {seeded_digest}")
    check("40 tasks in 20 colour-reversed pairs",
          len(tasks) == 40 and len({t["pair_id"] for t in tasks}) == 20)
    check("seeds are POSITIONAL (row i carries lo + i)",
          [t["seed"] for t in tasks] == list(range(lo, hi)))
    check("the SEEDED digest matches its pin", seeded_digest == RULES.SEEDED_TASK_DIGEST)
    check("and it is NOT the unseeded pin -- seeds change the full-field digest",
          RULES.SEEDED_TASK_DIGEST != RULES.TASK_DIGEST)
    try:
        summary = RUN.check_schedule(tasks)
        check("the runner ADMITS the registered schedule", True,
              f"pairs={summary['pairs']}")
    except Exception as e:                                    # noqa: BLE001
        check("the runner ADMITS the registered schedule", False, str(e)[:110])
    try:
        REF.validate_schedule_executable(tasks)
        check("the registry admits the schedule for EXECUTION", True)
    except Exception as e:                                    # noqa: BLE001
        check("the registry admits the schedule for EXECUTION", False, str(e)[:110])

    print("\n== the REAL builder, on the REGISTERED tasks ==")
    print("  (the call that aborted H2's attempt 1; stub evaluator, no JVM)")
    from . import eval_readout as RO
    from . import twixtbot_g3_reference as G3

    class _StubEvaluator:
        _g3_reference = "calib020_0001"
        _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"

    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": H2R.SELECTION_MODE})
    check("the frozen config is NOT already argmax, so the override is real",
          cfg.selection_mode != "argmax", f"frozen={cfg.selection_mode!r}")
    failures, arms, seeds_seen = [], set(), set()
    for t in tasks:
        task = dict(t, reference="calib020_0001",
                    reference_sha1="209cf2d4fd24a48553d259dd71b4954867b9473e")
        try:
            agent = G3.build_reference_agent(
                task=task, evaluator=_StubEvaluator(),
                colour=task["incumbent_colour"], config=argmax, capture=True)
            if not (agent.readout.mode == RO.MODE_ARGMAX
                    and agent.config.mcts_sims == H2R.MCTS_SIMS
                    and agent.config.board_size == 24
                    and agent.seed == task["seed"]):
                failures.append((task["task_id"], "built, configuration differs"))
            arms.add(task["incumbent_colour"])
            seeds_seen.add(agent.seed)
        except Exception as e:                                # noqa: BLE001
            failures.append((task["task_id"], f"{type(e).__name__}: {e}"))
    check(f"all {len(tasks)} REGISTERED tasks construct through the real builder",
          not failures, "" if not failures else f"{len(failures)} failed: {failures[:2]}")
    check("BOTH colour assignments were built", arms == {"red", "black"},
          f"arms={sorted(arms)}")
    check("every agent carries its own REGISTERED seed",
          seeds_seen == set(range(lo, hi)), f"{len(seeds_seen)} distinct seeds")
    check("the anchor and the incumbent's colour agree on every task",
          all(REF.reference_colour(t) == t["incumbent_colour"] for t in tasks))

    print("\n== the three outputs must be UNUSED ==")
    outs = (CMD.DEFAULT_RESULTS, CMD.DEFAULT_TRACE, CMD.DEFAULT_REPORT)
    for p in outs:
        print(f"  {'PRESENT' if os.path.lexists(p) else 'absent ':8s} {p}")
    check("all three are DISTINCT files", len(set(outs)) == 3)
    check("all three are ABSENT", not any(os.path.lexists(p) for p in outs))
    check("none is inside a SPENT run's directory",
          not any(p.startswith(d.rstrip("/") + "/")
                  for p in outs for d in CMD.SPENT_OUT_DIRS))
    check("the output DIRECTORY exists (nothing creates it at run time)",
          os.path.isdir(CMD.OUT_DIR), CMD.OUT_DIR)
    try:
        RUN.check_output_paths(*outs)
        check("check_output_paths ACCEPTS them", True)
    except Exception as e:                                    # noqa: BLE001
        check("check_output_paths ACCEPTS them", False, str(e)[:90])

    print("\n== the wrapper: gate restoration and child cleanup ==")
    print("  exit codes: 0 completed | 3 VOID | 4 unexpected | 5 unauthorized |")
    print("              6 timeout | 7 refused | 8 cleanup failed | 9 interrupted |")
    print("              10 GATE NOT RESTORED | 13 PARTIAL | 14 STOP RULE FIRED")
    check("restore_gate() is True on the already-closed real source",
          CMD.restore_gate() is True)
    check("the parser has NO --runner-source flag",
          "--runner-source" not in CMD._parser().format_help())
    check("no environment variable reaches the worker path",
          "os.environ" not in open(CMD.__file__, encoding="utf-8").read())
    import contextlib
    import io
    _err = io.StringIO()
    with contextlib.redirect_stderr(_err):
        code = CMD.main([])                 # spawns nothing, writes nothing
    print(f"  the refusal it printed: {_err.getvalue().strip()[:96]}")
    check("main() with the gate shut returns EXIT_UNAUTHORIZED",
          code == CMD.EXIT_UNAUTHORIZED, f"got {code}")
    check("and the gate is STILL closed afterwards",
          RUN.H3_PILOT_EXECUTION_AUTHORIZED is False)
    src = open(CMD.__file__, encoding="utf-8").read()
    check("restore_gate is called in main()'s finally and sets the code there",
          "restore_gate(_runner_source)" in src.split("    finally:")[-1]
          and "EXIT_GATE_NOT_RESTORED" in src.split("    finally:")[-1])
    check("the supervisor maps a surviving descendant to EXIT_CLEANUP_FAILED",
          'r["group_cleared"]' in src and "EXIT_CLEANUP_FAILED" in src)
    check("a PARTIAL run and a FIRED rule are RESULTS with their own codes",
          CMD.EXIT_PARTIAL == 13 and CMD.EXIT_STOP_RULE_FIRED == 14)

    print("\n== bounds and the stop rules, frozen before play ==")
    print(f"  whole-run limit {RULES.RUN_DEADLINE_S}s (CHOSEN, and it MAY time out)")
    print(f"  per-call bound  {RULES.PER_CALL_TIMEOUT_S}s")
    print(f"  S1 duplicate_pairs > {RULES.MAX_DUPLICATE_PAIRS} | "
          f"S2 caps > {RULES.MAX_CAPPED_GAMES} | "
          f"S3 within-pair identical > {RULES.MAX_WITHIN_PAIR_IDENTICAL}")
    print(f"  S4a total > {RULES.MAX_TOTAL_ELAPSED_S}s | "
          f"S4b p90 > {RULES.MAX_P90_OVER_MEDIAN}x median | "
          f"floor {RULES.REPORT_FLOOR_PAIRS} pairs")
    check("the analysis refuses a report with no per-game timing",
          _refuses_untimed())

    print("\n  the tests that BIND these claims, run here rather than cited:")
    r = subprocess.run([sys.executable, "-m", "pytest", *BINDING_TESTS, "-q",
                        "--tb=line", "-p", "no:cacheprovider"],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    last = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-1]
    print(f"    {last}")
    check("every binding test passes", r.returncode == 0)

    print("\n" + "=" * 74)
    print("⚠ RISK RECORDED BEFORE THE RUN, NOT AFTER. THE AUTHORIZED LAUNCH PATH HAS")
    print("  NEVER COMPLETED A PILOT. H2's attempt 3 proved the path end to end and")
    print("  then VOIDED on its own deadline; this design is new, its runner has")
    print("  never played a game, and the pilot MAY TIME OUT BY CONSTRUCTION -- 7,200 s")
    print("  is a CHOSEN limit, not a bound. A timeout is a reportable outcome, not a")
    print("  failure; an exception is a VOID and retires the block whole. Proceeding")
    print("  accepts both knowingly.")
    print("  IT CANNOT PRODUCE A STRENGTH VERDICT. Its outcomes are 'authorize design")
    print("  work on a full study' or 'close H3'.")
    failed = [lab for lab, ok in _checks if not ok]
    print(f"\n{len(_checks) - len(failed)}/{len(_checks)} checks PASS")
    if failed:
        print("🔴 FAILED CHECKS -- the pilot must NOT be authorized:")
        for lab in failed:
            print(f"    {lab}")
    return 0 if not failed else 1


def _refuses_untimed() -> bool:
    """The analysis must refuse a record with no duration -- H2's records had none,
    which is why that is a refusal and not a default."""
    try:
        ANALYSIS.summarise([{"task_id": "x", "pair_id": 0, "incumbent_colour": "red",
                             "transcript_digest": "a" * 64, "terminal_reason": "win",
                             "winner": "red", "t1j_points": 1.0, "plies": 10}],
                           total_elapsed_s=1.0)
    except ANALYSIS.H3AnalysisError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
