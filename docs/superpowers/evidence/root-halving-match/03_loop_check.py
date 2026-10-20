"""ROOT-HALVING MATCH -- LOOP EQUIVALENCE CHECK (CPU, no model). Does 01_match.py's copied game loop reproduce
eval_runner's? Ordinary vs ordinary through both: eval_runner._play_and_build_result and 01_match.play_and_build
(classes None, so mcts.MCTS for both colours).

Two games (one colour-reversed pair from eval_runner's agent builder) on test-only seeds TEST_ONLY_SEED_INTERVALS[0][0]
and the next; identical GPU-free fake evaluators (tests/eval_fakes.FakeEvaluator: uniform priors, value 0), a fresh
one per side per run; Stage B's PROTOCOL with 16 simulations per move (24x24, 280-ply cap, its readout).
PASS iff for every game the two results are identical (every field but replay_path: moves, winner, termination,
scores, binding) and the two replay files are identical as JSON. Any difference -> FAIL, reported, never relaxed.

  --out DIR (exists, without record.json)
Exit 0 COMPLETED (answer in the record) / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 110` (120 s in all).
"""
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIMS = 16
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def main(rec, out):
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero.search_distillation_stage_b import PROTOCOL
    from tests.eval_fakes import FakeEvaluator

    spec = importlib.util.spec_from_file_location("root_halving_match", HERE / "01_match.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    rec["scripts"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "01_match.py",
                                                                                    pathlib.Path(__file__))}
    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if n.split(".")[0] in ("scripts", "tests") and getattr(m, "__file__", None)}
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}

    seed = REF.TEST_ONLY_SEED_INTERVALS[0][0]
    if not (REF.seed_is_test_only(seed) and REF.seed_is_test_only(seed + 1)):
        raise Refused("the check's seeds are not test-only")
    cfg = ER.EvalConfig(**{**PROTOCOL, "mcts_sims": SIMS})
    if (cfg.board_size, cfg.max_moves, cfg.mcts_sims) != (24, 280, SIMS):
        raise Refused("the check is not 24x24 / 280 plies / 16 simulations")
    rd = ER.readout_from_eval_config(cfg)
    a, b = ER.AgentSpec("ordinary-a", "fake-a", rd), ER.AgentSpec("ordinary-b", "fake-b", rd)
    tasks = ER.build_agent_pairing_tasks("loop-check", a, b, 2, base_seed=seed)

    rec["games"] = []
    for task in tasks:
        runs = {}
        for name, play in (("existing", ER._play_and_build_result), ("copied", M.play_and_build)):
            t = time.monotonic()
            r = play(task, FakeEvaluator(), FakeEvaluator(), cfg, True, str(out / name))
            replay = json.loads(pathlib.Path(r.replay_path).read_text())   # relative to the CWD
            runs[name] = ({k: v for k, v in dataclasses.asdict(r).items() if k != "replay_path"}, replay,
                          round(time.monotonic() - t, 3))
        (ra, pa, sa), (rb, pb, sb) = runs["existing"], runs["copied"]
        rec["games"].append({"game_idx": task.game_idx, "seed": task.seed, "red": task.red_agent.agent_id,
                             "winner": ra["winner"], "reason": ra["reason"], "n_moves": ra["n_moves"],
                             "results_identical": ra == rb, "replays_identical": pa == pb,
                             "differing_result_fields": sorted(k for k in ra if ra[k] != rb.get(k)),
                             "seconds": {"existing": sa, "copied": sb}})
    rec["answer"] = "PASS" if all(g["results_identical"] and g["replays_identical"] for g in rec["games"]) else "FAIL"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "root-halving-loop-check/1", "script": "docs/superpowers/evidence/root-halving-match/03_loop_check.py",
           "head": git("rev-parse", "HEAD"), "simulations": SIMS}
    t0 = time.monotonic()
    try:
        verdict = main(rec, out)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Exception:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        verdict, rec["reason"] = "STOP_ERROR", traceback.format_exc()
    rec["verdict"], rec["seconds"] = verdict, round(time.monotonic() - t0, 3)
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if verdict == "COMPLETED":
        g = rec["games"]
        print(f"COMPLETED: {rec['answer']}; " + "; ".join(
            f"game {x['game_idx']} {x['reason']} {x['winner']} {x['n_moves']} plies, results "
            f"{'=' if x['results_identical'] else 'DIFFER'}, replays {'=' if x['replays_identical'] else 'DIFFER'}"
            for x in g) + f"; {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
