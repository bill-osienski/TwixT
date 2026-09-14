#!/usr/bin/env python3
"""H2 ATTEMPT 3 -- THE PRE-RUN VERIFICATION. READ-ONLY, and it is a STOP.

    .venv/bin/python -m scripts.GPU.alphazero.h2_prerun_verification

Measures the state a match authorization would be given against, and exits
NONZERO if any check fails. A verification that prints FAIL and returns 0 is the
defect this programme spent 2026-09-12 removing from its own harness.

WHAT IT DOES NOT DO. No gate is opened, no seed is drawn, no JVM starts, no model
is loaded, no game is played and nothing is written. The one effectful-looking
step is `build_reference_agent` on real scheduled tasks with a STUB evaluator --
the call that aborted on attempt 1 -- which constructs our own agent object and
touches neither T1j nor a seed's RNG stream. `restore_gate` is exercised on the
real runner source, where the gate is ALREADY closed, so it rewrites nothing.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from . import e4_screen_reference as REF
from . import h2_match_command as CMD
from . import h2_match_plan as PLAN
from . import h2_match_rules as R
from . import h2_match_runner as RUN

#: The commit whose work this verifies: the harness repairs and the third block's
#: registration. HEAD must DESCEND from it with a CLEAN tree -- asserting HEAD
#: EQUALS it would be circular, because this file and the output-destination fix it
#: found are themselves commits on top. A recorded HEAD plus a clean tree is the
#: binding; the ancestor names what is being built on.
BOUND_COMMIT = "c58b13d"

#: Every gate in the programme, by module attribute. All seven must be False.
GATES: Tuple[Tuple[str, str], ...] = (
    ("d1_probe", "D1_EXECUTION_AUTHORIZED"),
    ("e4_screen_command", "SCREEN_AUTHORIZED"),
    ("h1_viability_runner", "H1_EXECUTION_AUTHORIZED"),
    ("l0_match_command", "L0_EXECUTION_AUTHORIZED"),
    ("lowply_qualification", "LOWPLY_QUALIFICATION_AUTHORIZED"),
    ("runtime_requalification", "RUNTIME_REQUAL_AUTHORIZED"),
    ("h2_match_runner", "H2_EXECUTION_AUTHORIZED"),
)

#: The wrapper tests that bind the gate-restoration and child-cleanup claims. They
#: are RUN here rather than cited: a named test that does not pass proves nothing.
WRAPPER_TESTS = (
    "test_the_wrapper_refuses_with_the_shut_gate_and_verifies_it_closed",
    "test_the_wrapper_has_NO_runner_source_flag",
    "test_a_failed_restoration_becomes_the_wrappers_OWN_exit_code",
    "test_THE_FINALLY_PATH_also_restores_and_reports_its_own_failure",
    "test_the_finally_path_restores_a_REAL_open_gate_after_a_refusal",
    "test_restore_gate_is_FALSE_when_the_READBACK_disagrees",
    "test_the_worker_is_SUPERVISED_in_its_own_group_under_an_OUTER_cap",
    "test_every_outcome_gets_ITS_OWN_exit_code",
    "test_WORKER_REFUSES_without_the_supervisors_CAPABILITY",
    "test_THE_OUTPUT_DESTINATION_IS_NOT_A_SPENT_ATTEMPTS_DIRECTORY",
    "test_ALL_THREE_OUTPUTS_are_preflighted_before_any_game",
    "test_THE_CLEANUP_RUNS_AFTER_EVERY_GAME_including_a_failed_one",
)

_checks: List[Tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    _checks.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '🔴 FAIL':8s} {label}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print(f"H2 ATTEMPT 3 PRE-RUN VERIFICATION -- bound to {BOUND_COMMIT}")
    print("Nothing has been executed. The gate is shut at this moment.")
    print("=" * 74)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines()
    # 🔑 THE OUTPUT DIRECTORY IS THIS TOOL'S OWN PRODUCT and cannot be part of the
    # cleanliness it reports -- writing the record would otherwise dirty the tree it
    # is checking, and the check could never pass. Everything that is CODE must
    # match the commit exactly; nothing under the output directory affects behaviour.
    ignored = [l for l in porcelain if l[3:].strip().startswith(CMD.OUT_DIR)]
    dirty = "\n".join(l for l in porcelain if l not in ignored).strip()
    print("\n== the tree ==")
    descends = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BOUND_COMMIT, "HEAD"]).returncode == 0
    print(f"  HEAD {head}, descends from {BOUND_COMMIT}")
    check(f"HEAD descends from the bound commit ({BOUND_COMMIT})", descends)
    if ignored:
        print(f"  ignoring {len(ignored)} path(s) under {CMD.OUT_DIR} "
              f"(this tool's own output)")
    check("no SOURCE differs from the commit", not dirty,
          "" if not dirty else f"{len(dirty.splitlines())} path(s): "
                               f"{[l[3:] for l in dirty.splitlines()][:3]}")

    print("\n== the SEVEN gates ==")
    import importlib
    for mod_name, attr in GATES:
        mod = importlib.import_module(f"scripts.GPU.alphazero.{mod_name}")
        check(f"{attr:34s} False", getattr(mod, attr) is False)

    print("\n== the seed block ==")
    lo, hi = R.H2_SEED_BLOCK
    states = [REF.seed_status(s) for s in range(lo, hi)]
    acc = sum(s["accounted"] for s in states)
    exp = sum(s["exposed"] for s in states)
    ret = sum(s["retired"] for s in states)
    tst = sum(s["test_only"] for s in states)
    print(f"  [{lo}, {hi}): accounted {acc} / exposed {exp} / retired {ret} / "
          f"test_only {tst}")
    check("the block is 736 seeds", hi - lo == R.N_GAMES == 736)
    check("every seed ACCOUNTED", acc == 736)
    check("NOT exposed, NOT retired, NOT test-only -- a reservation is not a draw",
          (exp, ret, tst) == (0, 0, 0))
    check("no seed is in CONSUMED_SEEDS",
          not any(s in REF.CONSUMED_SEEDS for s in range(lo, hi)))
    try:
        RUN.check_seed_registration()
        check("the registration barrier is SATISFIED", True)
    except Exception as e:                                    # noqa: BLE001
        check("the registration barrier is SATISFIED", False, str(e)[:90])
    check("it is NOT either spent block",
          R.H2_SEED_BLOCK not in (R.H2_ATTEMPT1_SEED_BLOCK, R.H2_ATTEMPT2_SEED_BLOCK))

    print("\n== the frozen schedule ==")
    tasks = PLAN.build_tasks(PLAN.load_source_plan())
    summary = PLAN.validate_h2_schedule(tasks)
    design = R.L0.l0_task_digest(tasks)
    full = R.h2_full_task_digest(tasks)
    print(f"  {len(tasks)} tasks, {summary['cells']} cells x "
          f"{summary['reps_per_cell']} reps, mode {summary['selection_mode']}")
    print(f"  design digest     : {design}")
    print(f"  full-field digest : {full}")
    check("736 tasks", len(tasks) == 736)
    check("16 cells x 46 reps", (summary["cells"], summary["reps_per_cell"]) == (16, 46))
    check("the design digest matches the pin", design == R.H2_TASK_DIGEST)
    check("the FULL-FIELD digest matches the pin", full == R.H2_FULL_TASK_DIGEST)
    check("neither digest is a previous attempt's",
          design not in (R.H2_ATTEMPT1_TASK_DIGEST, R.H2_ATTEMPT2_TASK_DIGEST))
    check("seeds are POSITIONAL (row i carries lo + i)",
          [t["seed"] for t in tasks] == list(range(lo, hi)))
    check("every task carries selection_mode argmax",
          all(t.get("selection_mode") == "argmax" for t in tasks))
    try:
        REF.validate_schedule_executable(tasks)
        check("the registry admits this schedule for EXECUTION", True)
    except Exception as e:                                    # noqa: BLE001
        check("the registry admits this schedule for EXECUTION", False, str(e)[:90])

    print("\n== the ACTUAL argmax configuration, through the REAL builder ==")
    print("  (the call that aborted at ply 7 on attempt 1; stub evaluator, no JVM)")
    from . import eval_readout as RO
    from . import twixtbot_g3_reference as G3

    class _StubEvaluator:
        _g3_reference = "calib020_0001"
        _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"

    cfg = G3.eval_config()
    argmax = cfg.__class__(**{**cfg.__dict__, "selection_mode": R.SELECTION_MODE})
    check("the frozen config is NOT already argmax, so the override is real",
          cfg.selection_mode != "argmax", f"frozen={cfg.selection_mode!r}")
    built: Dict[Any, Any] = {}
    first_per_cell = {}
    for t in tasks:
        first_per_cell.setdefault((t["opening"], t["colour_arm"]), t)
    failures = []
    for key, t in first_per_cell.items():
        try:
            agent = G3.build_reference_agent(
                task=t, evaluator=_StubEvaluator(), colour=REF.reference_colour(t),
                config=argmax, capture=True)
            built[key] = agent
            if not (agent.readout.mode == RO.MODE_ARGMAX
                    and agent.config.selection_mode == "argmax"
                    and agent.config.mcts_sims == R.MCTS_SIMS
                    and agent.config.board_size == 24
                    and agent.seed == t["seed"]):
                failures.append((key, "constructed but the configuration differs"))
        except Exception as e:                                # noqa: BLE001
            failures.append((key, f"{type(e).__name__}: {e}"))
    check(f"all {len(first_per_cell)} cells' first task construct "
          f"(both colour arms, eight openings)", not failures,
          "" if not failures else f"{len(failures)} failed: {failures[:2]}")
    t0 = tasks[0]
    a0 = built.get((t0["opening"], t0["colour_arm"]))
    if a0 is not None:
        print(f"  task 0: seed {a0.seed} arm {t0['colour_arm']} anchor "
              f"{t0['anchor_colour']} our colour {REF.reference_colour(t0)}")
        print(f"          readout {a0.readout.mode!r} sims {a0.config.mcts_sims} "
              f"board {a0.config.board_size}")
        check("task 0's agent carries the SCHEDULED seed", a0.seed == lo == 202622000)
        check("task 0's readout mode is ARGMAX", a0.readout.mode == RO.MODE_ARGMAX)
    else:
        check("task 0 constructed", False)
    arms = {k[1] for k in built}
    check("BOTH colour arms were constructed", arms == {"t1j_red", "t1j_black"},
          f"arms={sorted(arms)}")

    print("\n== the incumbent identity H2 will record ==")
    ident = RUN.frozen_incumbent_identity()
    for k in sorted(ident):
        print(f"  {k:20s}: {ident[k]}")
    check("the identity's eval_config records selection_mode argmax",
          ident.get("eval_config", {}).get("selection_mode") == "argmax",
          f"{ident.get('eval_config', {}).get('selection_mode')!r}")
    check("it records mcts_sims 400 on a board of 24",
          (ident.get("eval_config", {}).get("mcts_sims"),
           ident.get("eval_config", {}).get("board_size")) == (R.MCTS_SIMS, 24))
    check("the readout path is eval_readout.select, not mcts.select_move",
          "eval_readout.select" in str(ident.get("readout_path", "")))
    check("the identity pins the source plan's digest",
          len(str(ident.get("plan_sha256", ""))) == 64)
    check("the temperature settings are recorded as INERT, not as unchanged",
          sorted(ident.get("inert_under_argmax", {})) ==
          sorted(R.INERT_UNDER_ARGMAX),
          f"{ident.get('inert_under_argmax')}")
    try:
        RUN.check_incumbent_identity(ident)
        check("the runner ACCEPTS its own frozen identity", True)
    except Exception as e:                                    # noqa: BLE001
        check("the runner ACCEPTS its own frozen identity", False, str(e)[:90])

    print("\n== the toolchain ==")
    try:
        from . import t1j_toolchain as TC
        paths = TC.verified_paths()
        jar = paths["jar"] if isinstance(paths, dict) else paths[0]
        print(f"  jar {jar}")
        print(f"  jar sha256 {_sha256(jar)}")
        check("the jar resolves and is readable", os.path.isfile(jar))
    except Exception as e:                                    # noqa: BLE001
        check("the toolchain resolves", False, f"{type(e).__name__}: {e}")

    print("\n== the three outputs must be UNUSED ==")
    outs = (CMD.DEFAULT_RESULTS, CMD.DEFAULT_TRACE, CMD.DEFAULT_REPORT)
    for p in outs:
        print(f"  {'PRESENT' if os.path.lexists(p) else 'absent ':8s} {p}")
    check("all three are DISTINCT files", len(set(outs)) == 3)
    check("all three are ABSENT", not any(os.path.lexists(p) for p in outs))
    check("none is inside a SPENT attempt's directory",
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
    print("              10 GATE NOT RESTORED | 11 degenerate | 12 no rate")
    check("restore_gate() is True on the already-closed real source",
          CMD.restore_gate() is True)
    check("the parser has NO --runner-source flag",
          "--runner-source" not in CMD._parser().format_help())
    check("no environment variable reaches the worker path",
          "environ" not in open(CMD.__file__, encoding="utf-8").read().split(
              "def worker_main")[1].split("def _persist_and_classify")[0])
    # THE UNAUTHORIZED PATH, EXERCISED LIVE. It spawns nothing and writes nothing.
    # Its refusal goes to stderr; captured so it does not interleave into this record.
    import contextlib
    import io
    _err = io.StringIO()
    with contextlib.redirect_stderr(_err):
        code = CMD.main([])
    print(f"  the refusal it printed: {_err.getvalue().strip()[:100]}")
    check("main() with the gate shut returns EXIT_UNAUTHORIZED",
          code == CMD.EXIT_UNAUTHORIZED, f"got {code}")
    check("and the gate is STILL closed afterwards",
          RUN.H2_EXECUTION_AUTHORIZED is False)
    # a failed restoration SUPERSEDES every other code -- structural, because
    # exercising it would require opening the real gate
    src = open(CMD.__file__, encoding="utf-8").read()
    tail = src.split("    finally:")[-1]
    check("restore_gate is called in main()'s finally, and sets the code there",
          "restore_gate(_runner_source)" in tail and "EXIT_GATE_NOT_RESTORED" in tail)
    check("the supervisor checks group_cleared and maps it to EXIT_CLEANUP_FAILED",
          'r["group_cleared"]' in src and "EXIT_CLEANUP_FAILED" in src)

    print("\n  the tests that BIND those claims, run here rather than cited:")
    nodes = [f"tests/test_h2_match.py::{t}" for t in WRAPPER_TESTS]
    r = subprocess.run([sys.executable, "-m", "pytest", *nodes, "-q", "-rf",
                        "--tb=line", "-p", "no:cacheprovider"],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    last = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-1]
    print(f"    {last}")
    check(f"all {len(WRAPPER_TESTS)} wrapper tests pass", r.returncode == 0)

    print("\n== bounds ==")
    print(f"  runner deadline   {RUN.RUN_DEADLINE_S}s "
          f"({RUN.RUN_DEADLINE_S // 60} min)")
    print(f"  supervisor cap    {RUN.RUN_DEADLINE_S + CMD.SUPERVISOR_GRACE_S}s "
          f"(grace {CMD.SUPERVISOR_GRACE_S}s)")
    print(f"  interrupt grace   {CMD.INTERRUPT_GRACE_S}s")
    check("the supervisor cap EXCEEDS the runner deadline, so a deadline VOID is "
          "reported as a VOID rather than killed", CMD.SUPERVISOR_GRACE_S > 0)

    print("\n" + "=" * 74)
    print("⚠ RISK RECORDED BEFORE THE RUN, NOT AFTER. The production seam has still")
    print("  never played a game: construction, wiring and refusals are tested, and")
    print("  the only two things that ever reached a board were attempt 1 (VOID at")
    print("  task 0) and the 383 games an unauthorized control played. If it fails at")
    print("  game 1 the run VOIDs and the one-shot rule retires all 736 seeds -- the")
    print("  same cost as a failure at game 735. Proceeding accepts that knowingly.")
    failed = [lab for lab, ok in _checks if not ok]
    print(f"\n{len(_checks) - len(failed)}/{len(_checks)} checks PASS")
    if failed:
        print("🔴 FAILED CHECKS -- the match must NOT be authorized:")
        for lab in failed:
            print(f"    {lab}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
