"""H3 FULL STUDY — the pre-run verification. READ-ONLY, AND IT IS A STOP.

Exits non-zero if any check fails. It resolves the real objects and asks the real
questions; it never patches in what the thing under test is missing, because the
pilot's did exactly that and reported 40 tasks building through a builder that
would have refused every one of them.

🔴 IT IS EXPECTED TO FAIL TODAY, and that is the point: the opening set has not
been generated and no seed block is reserved, so the checks that depend on them
report NOT READY rather than passing vacuously.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from . import gate_inventory as INVENTORY
from . import e4_screen_reference as REF
from . import h3_study_analysis as ANALYSIS
from . import h3_study_command as CMD
from . import h3_study_generator as GEN
from . import h3_study_rules as RULES
from . import h3_study_runner as RUN

#: The commit whose work this verifies. HEAD must DESCEND from it with a clean
#: SOURCE tree; asserting equality would be circular.
BOUND_COMMIT = "f691294"

#: 🔴 DERIVED FROM SOURCE, never hand-kept. This module's own list said
#: the wrong number for weeks; see `gate_inventory` for why.
GATES: Tuple[Tuple[str, str], ...] = INVENTORY.gates()

_FAILED: List[str] = []
_NOT_READY: List[str] = []


def check(label: str, ok: Any, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL':8s} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _FAILED.append(label)
    return bool(ok)


def pending(label: str, detail: str = "") -> None:
    """A precondition that is ABSENT BY DESIGN — not a failure, and not a pass.

    🔑 A THIRD OUTCOME ON PURPOSE. Reporting "the seeds are registered: FAIL"
    would read as a defect; reporting it as PASS would be a lie. The study is not
    ready to run and the verification says so in its own word.
    """
    print(f"  {'PENDING':8s} {label}" + (f"  {detail}" if detail else ""))
    _NOT_READY.append(label)


def main() -> int:
    print(f"H3 FULL STUDY PRE-RUN VERIFICATION -- bound to {BOUND_COMMIT}")
    print("Nothing has been executed. Both gates are shut at this moment.")
    print("=" * 74)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines()
    dirty = "\n".join(porcelain).strip()
    descends = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BOUND_COMMIT, "HEAD"]).returncode == 0

    print("\n== the tree ==")
    print(f"  HEAD {head}, descends from {BOUND_COMMIT}")
    check(f"HEAD descends from the bound commit ({BOUND_COMMIT})", descends)
    check("no SOURCE differs from the commit", not dirty,
          "" if not dirty else f"{[l[3:] for l in dirty.splitlines()][:3]}")

    print("\n== the TEN gates ==")
    import importlib
    for mod_name, attr in GATES:
        mod = importlib.import_module(f"scripts.GPU.alphazero.{mod_name}")
        check(f"{attr:34s} False", getattr(mod, attr) is False)
    check("the study and generation gates are SEPARATE constants",
          "H3_GENERATION_AUTHORIZED" not in open(RUN.__file__, encoding="utf-8").read()
          and "H3_STUDY_EXECUTION_AUTHORIZED" not in open(GEN.__file__,
                                                          encoding="utf-8").read())

    print("\n== the design's arithmetic, recomputed here ==")
    import math
    h = RULES.half_width(RULES.N_PAIRS)
    print(f"  N {RULES.N_PAIRS} pairs / {RULES.N_GAMES} games | h = {h:.5f}")
    check("h <= the declared precision target",
          h <= RULES.PRECISION_TARGET, f"{h:.5f} <= {RULES.PRECISION_TARGET}")
    check("288 would NOT have met it (the Amendment 1 correction)",
          RULES.half_width(288) > RULES.PRECISION_TARGET,
          f"{RULES.half_width(288):.5f}")
    check("the ceiling of the requirement is 289",
          math.ceil(math.log(2 / RULES.ALPHA)
                    / (2 * RULES.PRECISION_TARGET ** 2)) == 289)
    check("every balance divides without remainder",
          RULES.PAIRS_PER_SEGMENT * RULES.N_SEGMENTS == RULES.N_PAIRS
          and RULES.GAMES_PER_SEGMENT * RULES.N_SEGMENTS == RULES.N_GAMES)
    check("ONE population, and nothing survives of the second",
          RULES.STRATA == (RULES.STRATUM_UNIFORM,)
          and not any(hasattr(RULES, a) for a in
                      ("STRATUM_CO_PRODUCED", "PAIRS_PER_STRATUM",
                       "STRATUM_PAIRS_PER_SEGMENT", "INCUMBENT_FIRST_PER_SEGMENT",
                       "T1J_FIRST_PER_SEGMENT", "ORDER_INCUMBENT_FIRST",
                       "ORDER_T1J_FIRST", "generation_config")))

    print("\n== generation is ENGINE-FREE ==")
    check("the population writer has NO execution gate",
          not hasattr(GEN, "H3_GENERATION_AUTHORIZED"),
          "no engine runs, so there is nothing to gate")
    check("it has a POPULATION-FREEZE barrier instead",
          GEN.H3_POPULATION_FREEZE_AUTHORIZED is False)
    check("the generator declares itself engine-free",
          GEN.generator_identity()["engine_free"] is True)
    check("no active study module reaches the retired engine path",
          not any("coproduced" in getattr(m, "__name__", "")
                  for m in (RULES, GEN, ANALYSIS, RUN)))

    print("\n== the generation seeds: declared, and in NO registry ==")
    lo, hi = RULES.generation_seed_range(RULES.GEN_SEED_UNIFORM)
    bad = [x for x in (lo, (lo + hi) // 2, hi - 1)
           if any(REF.seed_status(x).values()) or x in REF.CONSUMED_SEEDS]
    check(f"the range [{lo}, {hi}) is in no registry", not bad, str(bad))
    check(f"it covers every candidate: {RULES.N_PAIRS} x {RULES.MAX_ATTEMPTS} "
          f"= {RULES.N_PAIRS * RULES.MAX_ATTEMPTS}",
          hi - lo == RULES.N_PAIRS * RULES.MAX_ATTEMPTS)
    #: 🔴 THE THREE RETIRED RANGES ARE IN NO REGISTRY EITHER, so they are checked
    #: by hand. Skipping them is how an overlapping candidate reads as clean.
    for r_lo, r_hi, why in RULES.RETIRED_GENERATION_RANGES:
        check(f"clear of the retired range [{r_lo}, {r_hi})",
              hi <= r_lo or r_hi <= lo, why.split(".")[0])
        check(f"  ... and separated from it by the gap floor "
              f"({RULES.N_PAIRS * RULES.MAX_ATTEMPTS})",
              (lo - r_hi if lo >= r_hi else r_lo - hi)
              >= RULES.N_PAIRS * RULES.MAX_ATTEMPTS)
    check("every retired range is REFUSED by the spent check",
          all(RULES.is_spent_generation_range(r_lo)
              for r_lo, _, _ in RULES.RETIRED_GENERATION_RANGES))
    check("the live range is NOT refused",
          RULES.is_spent_generation_range(RULES.GEN_SEED_UNIFORM) is False)

    print("\n== THE population, generated here in memory ==")
    uniform = RULES.generate_uniform_openings()
    check(f"{RULES.N_PAIRS} uniform openings at ply {RULES.OPENING_PLIES}",
          len(uniform) == RULES.N_PAIRS)
    check("all distinct up to symmetry",
          len({o["digest"] for o in uniform}) == len(uniform))
    excluded = RULES.excluded_digests()
    check("none duplicates a PILOT or H1/H2 position",
          not {o["digest"] for o in uniform} & excluded,
          f"exclusion set holds {len(excluded)}")
    check("the exclusion set is NOT vacuous", len(excluded) == 28)
    print(f"  attempts: min {min(o['attempts'] for o in uniform)} "
          f"max {max(o['attempts'] for o in uniform)} "
          f"(recorded per opening, because the rejection rate conditions the set)")

    print("\n== the population writer: ENGINE-FREE, and its artifact schema ==")
    check("the artifact SCHEMA is frozen",
          set(GEN.ARTIFACT_KEYS) == {"design", "stratum", "n", "generation_note",
                                     "generator", "claim", "openings",
                                     "opening_set_digest"},
          str(sorted(GEN.ARTIFACT_KEYS)))
    check("NO generating-engine configuration is pinned, because none generates",
          "config_pins" not in GEN.ARTIFACT_KEYS
          and "selection_mode" not in GEN.ARTIFACT_KEYS)
    check("every opening must carry its ATTEMPT COUNT",
          "attempts" in GEN.OPENING_KEYS and "order" not in GEN.OPENING_KEYS)
    _bad_stratum = dict(_good_doc := GEN.artifact_document(GEN.build_population()))
    _bad_stratum["stratum"] = "co_produced"
    check("a CO-PRODUCED artifact is REFUSED",
          _refuses(lambda: GEN.validate_artifact(_bad_stratum), "stratum"))
    _short = dict(_good_doc); _short["n"] = 148
    check("a 148-opening artifact is REFUSED (wrong population size)",
          _refuses(lambda: GEN.validate_artifact(_short), "296"))
    _weak = dict(_good_doc); _weak["claim"] = _good_doc["claim"].replace("NOTHING", "little")
    check("a WEAKENED claim is REFUSED",
          _refuses(lambda: GEN.validate_artifact(_weak), "claim"))
    _ordered = dict(_good_doc)
    _ordered["openings"] = [{**_good_doc["openings"][0], "order": "incumbent_first"}] \
        + list(_good_doc["openings"][1:])
    check("a stale `order` field is REFUSED",
          _refuses(lambda: GEN.validate_artifact(_ordered), "order"))
    check("the good artifact passes",
          GEN.validate_artifact(_good_doc)["n"] == RULES.N_PAIRS)

    print("\n== the FROZEN population ==")
    check("the official destination is ABSENT", not os.path.lexists(GEN.OUT_DIR))
    check("freezing is BARRED, and a refusal creates nothing",
          _refuses(lambda: GEN.freeze_population(), "NOT AUTHORIZED")
          and not os.path.lexists(GEN.OUT_DIR))
    if os.path.lexists(RUN.OPENING_SET_PATH):
        check("the pinned opening set loads and passes its own checks",
              bool(RUN.load_opening_set(RUN.OPENING_SET_PATH)))
    else:
        pending("the population has been FROZEN",
                f"{RUN.OPENING_SET_PATH} does not exist -- freezing the "
                f"population is a separate authorized step")
    pending("OPENING_SET_DIGEST is pinned",
            "unset until the freeze step produces the artifact"
            if RULES.OPENING_SET_DIGEST is None else "")

    print("\n== the schedule, over the in-memory population ==")
    population = GEN.build_population()
    check("it passes check_opening_set",
          RULES.check_opening_set(population)["n"] == RULES.N_PAIRS)
    tasks = RULES.build_tasks(population)
    check(f"{RULES.N_GAMES} tasks in {RULES.N_PAIRS} pairs",
          len(tasks) == RULES.N_GAMES
          and len({t["pair_id"] for t in tasks}) == RULES.N_PAIRS)
    check("no seed is assigned", all(t["seed"] is None for t in tasks))
    check("every task carries stratum, segment and opening_digest",
          all(t["stratum"] == RULES.STRATUM_UNIFORM and t["segment"] in range(4)
              and len(t["opening_digest"]) == 64 for t in tasks))
    check("NO task carries an `order`", not any("order" in t for t in tasks))
    check("every task carries the reference identity the builder READS",
          all(t.get("reference") and t.get("reference_sha1")
              and t.get("reference_colour") == REF.reference_colour(t)
              for t in tasks))
    for k in range(RULES.N_SEGMENTS):
        seg = RUN.segment_schedule(tasks, k)
        check(f"segment {k}: {RULES.PAIRS_PER_SEGMENT} pairs x 2 arms = "
              f"{RULES.GAMES_PER_SEGMENT} games, one population",
              len(seg) == RULES.GAMES_PER_SEGMENT
              and len({t["pair_id"] for t in seg}) == RULES.PAIRS_PER_SEGMENT
              and {t["stratum"] for t in seg} == {RULES.STRATUM_UNIFORM})
    check("each segment's digest is its OWN",
          len({RUN.segment_digest(tasks, k)
               for k in range(RULES.N_SEGMENTS)}) == RULES.N_SEGMENTS)

    print("\n== the REAL builder, on the population's tasks ==")
    #: 🔴 THE SECOND DEAD REFERENCE. `match_cfg` was defined in the
    #: entropy-finding block, which went when generation became engine-free --
    #: but it is the PLAYING configuration and everything below still needs it.
    #: Both NameErrors survived because the crash killed the run before the
    #: verdict, and a checker that dies mid-way looks like one that finished.
    #: ONE OBJECT, constructed here and handed to the builder AND to the identity
    #: derivation, so the recorded identity describes the object that was used.
    match_cfg = RUN.frozen_argmax_config()
    from . import eval_readout as RO
    from . import twixtbot_g3_reference as G3

    class _StubEvaluator:
        _g3_reference = "calib020_0001"
        _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"

    check("the schedule's identity IS the registry's pin",
          REF.REFERENCE_CHECKPOINTS[_StubEvaluator._g3_reference]["sha1"]
          == _StubEvaluator._g3_sha1)
    #: 🔴 `stub` -- a NameError since the uniform rewrite renamed it to
    #: `population`. The preflight CRASHED here, so everything below this line
    #: (the seed block, the output paths, the wrapper, the VERDICT) never ran and
    #: the exit code was 1. Reading the tail showed PASS lines and nothing wrong.
    #: A checker that dies mid-way looks exactly like one that finished.
    seeded = RULES.build_tasks(population, seed_interval=(777000000,
                                                          777000000 + RULES.N_GAMES))
    missing = sorted({f for t in seeded
                      for f in ("reference", "reference_sha1", "reference_colour")
                      if f not in t})
    check("the SCHEDULED tasks already carry the identity fields, UNPATCHED",
          not missing, f"missing {missing}")
    failures, arms = [], set()
    for task in seeded[:32]:
        try:
            agent = G3.build_reference_agent(
                task=task, evaluator=_StubEvaluator(),
                colour=REF.reference_colour(task), config=match_cfg, capture=True)
            if not (agent.readout.mode == RO.MODE_ARGMAX
                    and agent.seed == task["seed"]):
                failures.append((task["task_id"], "configuration differs"))
            arms.add(task["incumbent_colour"])
        except Exception as e:                                # noqa: BLE001
            failures.append((task["task_id"], f"{type(e).__name__}: {e}"))
    check("32 tasks construct through the real builder", not failures,
          str(failures[:2]))
    check("BOTH colour assignments were built", arms == {"red", "black"})

    print("\n== the incumbent identity the record will carry ==")
    ident = RUN.frozen_incumbent_identity(match_cfg)
    print(f"  {ident['reference']} selection_mode "
          f"{ident['eval_config']['selection_mode']!r} sims "
          f"{ident['eval_config']['mcts_sims']}")
    check("the identity barrier ACCEPTS the frozen identity",
          _accepts(lambda: RUN.check_incumbent_identity(ident, match_cfg)))
    check("the recorded identity IS the config passed to the builder",
          not [f for f, v in ident["argmax_config"].items()
               if getattr(match_cfg, f) != v])
    bad_cfg = match_cfg.__class__(**{**match_cfg.__dict__, "mcts_sims": 401})
    check("and it REFUSES a DRIFTED config object (negative control)",
          _refuses(lambda: RUN.check_incumbent_identity(
              RUN.frozen_incumbent_identity(bad_cfg), bad_cfg), "disagrees"))

    print("\n== the seed block ==")
    if RUN.STUDY_SEED_BLOCK is None:
        pending("a seed block is reserved and registered",
                f"{RULES.N_GAMES} seeds, four 148-seed quarters, with a "
                f"collision re-proof covering the GENERATION ranges too")
    else:
        check("the block is registered", _accepts(RUN.check_seed_registration))

    print("\n== the outputs must be UNUSED ==")
    for k in range(RULES.N_SEGMENTS):
        d = RUN.segment_out_dir(k)
        paths = CMD.default_paths(k)
        check(f"segment {k}: all three absent",
              not any(os.path.lexists(p) for p in paths), d)
        check(f"segment {k}: not inside a SPENT run's directory",
              not any(d == s or d.startswith(s.rstrip('/') + '/')
                      for s in CMD.SPENT_OUT_DIRS))

    print("\n== the wrapper ==")
    check("restore_gate() is True on the already-closed real source",
          CMD.restore_gate() is True)
    check("the parser has NO --runner-source and no gate flag",
          all(f not in CMD._parser().format_help()
              for f in ("--runner-source", "--authorize", "--force")))
    check("--segment is REQUIRED", "--segment" in CMD._parser().format_help())
    check("no environment variable reaches the worker",
          "os.environ" not in open(CMD.__file__, encoding="utf-8").read())
    check("a PARTIAL run and a FIRED gate are RESULTS with their own codes",
          CMD.EXIT_PARTIAL == 13 and CMD.EXIT_STOP_RULE_FIRED == 14)

    print("\n== the analysis's verdict rule ==")
    check("the null is parity at 0.50", ANALYSIS.PARITY == 0.5)
    check("an UNCOMPUTABLE sensitivity suppresses the verdict",
          ANALYSIS.evaluate_verdict(
              primary={"decisive": True, "favours": "incumbent", "mean": 0.9,
                       "computable": True},
              sensitivities={"x": {"computable": False, "mean": None,
                                   "favours": None, "decisive": False}}
          )["verdict"] == ANALYSIS.NO_VERDICT)
    check("a sensitivity of EXACTLY 0.50 suppresses the verdict",
          ANALYSIS.evaluate_verdict(
              primary={"decisive": True, "favours": "incumbent", "mean": 0.9,
                       "computable": True},
              sensitivities={"x": {"computable": True, "mean": 0.5,
                                   "favours": None, "decisive": False}}
          )["verdict"] == ANALYSIS.NO_VERDICT)
    check("a sensitivity DECISIVE the other way suppresses the verdict",
          ANALYSIS.evaluate_verdict(
              primary={"decisive": True, "favours": "incumbent", "mean": 0.62,
                       "computable": True},
              sensitivities={"x": {"computable": True, "mean": 0.3,
                                   "favours": "t1j", "decisive": True}}
          )["verdict"] == ANALYSIS.NO_VERDICT)
    check("agreement DOES yield a verdict (so the rule is not an outage)",
          ANALYSIS.evaluate_verdict(
              primary={"decisive": True, "favours": "incumbent", "mean": 0.62,
                       "computable": True},
              sensitivities={"x": {"computable": True, "mean": 0.61,
                                   "favours": "incumbent", "decisive": True}}
          )["verdict"] == ANALYSIS.INCUMBENT_STRONGER)
    check("a WITHHELD segment after inspection blocks a verdict",
          RUN.combine_segments([
              {"segment": 0, "status": "completed", "outcomes_inspected": True},
              {"segment": 1, "status": "withheld", "outcomes_inspected": True},
          ])["verdict_permitted"] is False)
    check("a VOIDed segment blocks an automatic verdict",
          RUN.combine_segments([
              {"segment": 0, "status": "void", "outcomes_inspected": False},
          ])["verdict_permitted"] is False)

    print("\n  the tests that BIND these claims, run here rather than cited:")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_h3_study_rules.py",
         "tests/test_h3_study_analysis.py", "tests/test_h3_study_runner.py",
         "-p", "no:cacheprovider"], capture_output=True, text=True)
    print("    " + (r.stdout.strip().splitlines() or ["<no output>"])[-1])
    check("every binding test passes", r.returncode == 0)

    print("\n" + "=" * 74)
    print("⚠ SCOPE. This establishes that the study's machinery refuses, computes")
    print("  and constructs as the card says -- on this tree, at this commit, with")
    print("  every gate shut. It establishes NOTHING about strength, and the study")
    print("  CANNOT RUN: the co-produced stratum has not been generated and no seed")
    print("  block is reserved. Each is a separate authorization.")
    total = len(_FAILED) + len(_NOT_READY)
    print(f"\n{len(_FAILED)} FAILED | {len(_NOT_READY)} PENDING BY DESIGN")
    for label in _NOT_READY:
        print(f"    PENDING: {label}")
    for label in _FAILED:
        print(f"    🔴 FAILED: {label}")
    if _FAILED:
        return 1
    print("\nNo check FAILED. The study is NOT READY TO RUN, by design.")
    return 0


def _refuses(fn, needle: str = "") -> bool:
    try:
        fn()
    except Exception as e:                                    # noqa: BLE001
        return needle.lower() in str(e).lower() if needle else True
    return False


def _accepts(fn) -> bool:
    try:
        fn()
        return True
    except Exception:                                         # noqa: BLE001
        return False


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
