"""MIRROR AVERAGING + ROOT SEQUENTIAL HALVING -- THE STEP-B LOOP CHECKS (CPU, fake evaluators, no model, no GPU).
Card `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` §5 B.

(a) EQUIVALENCE, as entry 15's root-halving-match/03_loop_check.py: ordinary vs ordinary through both
    eval_runner._play_and_build_result and the shared copied loop mirror_halving_combination.play_and_build (classes
    None), one colour-reversed pair on test-only seeds TEST_ONLY_SEED_INTERVALS[0][0] and the next, tests/eval_fakes
    FakeEvaluator (uniform priors, value 0), 16 simulations, Stage B's PROTOCOL otherwise. PASS iff each game's results
    (every field but replay_path) and replay files are identical.
(b) THE COMBINATION: one colour-reversed pair on the same two test-only seeds, through the shared loop: the candidate
    (RootSequentialHalvingMCTS over MirrorAveragingEvaluator, 200 simulations) against the incumbent (MCTS, 400), each
    side over its own BoardFake -- priors and value computed from the board tensor and the move coordinates, NOT
    symmetric under the left-right reflection, so the averaging changes them (checked first on three positions).
    PASS iff both games complete; every candidate search from ply 20 supplied last_choice, and the move recorded at
    that ply is it; no candidate search before ply 20, and no incumbent search, supplied one. REPORTED, not gated:
    the plies where the readout (eval_readout.select on the same counts, its own test-only RNG) would have chosen a
    different move.

  --out DIR (exists, without record.json)
Exit 0 COMPLETED (answer in the record) / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 590` (600 s in all).
"""
import dataclasses
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIMS_A = 16
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


class BoardFake:
    """A GPU-free evaluator whose outputs depend on the board and the moves, and are not mirror-symmetric."""

    def __init__(self):
        import numpy as np
        self.np, self.network = np, None                             # MirrorAveragingEvaluator reads .network

    def build_input_tensor(self, state):
        return state.to_tensor().astype(self.np.float32)            # (C, H, W), the real 30-channel encoding

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        np = self.np
        occ = boards[..., :2].sum(axis=-1)                           # (B, H, W)
        cols = np.arange(boards.shape[2], dtype=np.float32)
        right = np.roll(occ, -1, axis=2)                             # the right neighbour's occupancy: asymmetric
        b_idx = np.arange(boards.shape[0])[:, None]
        logits = 0.08 * move_cols + 0.03 * move_rows + 1.5 * right[b_idx, move_rows, move_cols]
        logits = np.where(move_mask > 0, logits, -1e9)
        p = np.exp(logits - logits.max(axis=1, keepdims=True)) * (move_mask > 0)
        priors = (p / p.sum(axis=1, keepdims=True)).astype(np.float32)
        tilt = ((boards[..., 0] - boards[..., 1]) * (cols - 11.5)).sum(axis=(1, 2))
        return priors, np.tanh(0.02 * tilt).astype(np.float32)


def part_a(M, ER, PROTOCOL, FakeEvaluator, seed, out):
    cfg = ER.EvalConfig(**{**PROTOCOL, "mcts_sims": SIMS_A})
    if (cfg.board_size, cfg.max_moves, cfg.mcts_sims) != (24, 280, SIMS_A):
        raise Refused("part (a) is not 24x24 / 280 plies / 16 simulations")
    rd = ER.readout_from_eval_config(cfg)
    a, b = ER.AgentSpec("ordinary-a", "fake-a", rd), ER.AgentSpec("ordinary-b", "fake-b", rd)
    games = []
    for task in ER.build_agent_pairing_tasks("loop-check-a", a, b, 2, base_seed=seed):
        runs = {}
        for name, play in (("existing", ER._play_and_build_result), ("copied", M.play_and_build)):
            r = play(task, FakeEvaluator(), FakeEvaluator(), cfg, True, str(out / "a" / name))
            runs[name] = ({k: v for k, v in dataclasses.asdict(r).items() if k != "replay_path"},
                          json.loads(pathlib.Path(r.replay_path).read_text()))
        (ra, pa), (rb, pb) = runs["existing"], runs["copied"]
        games.append({"game_idx": task.game_idx, "seed": task.seed, "reason": ra["reason"], "n_moves": ra["n_moves"],
                      "results_identical": ra == rb, "replays_identical": pa == pb,
                      "differing_result_fields": sorted(k for k in ra if ra[k] != rb.get(k))})
    return {"games": games, "answer": "PASS" if all(g["results_identical"] and g["replays_identical"]
                                                    for g in games) else "FAIL"}


def part_b(M, ER, RSH, MCTS, MirrorAveragingEvaluator, PROTOCOL, seed, out):
    import numpy as np
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    cfg = ER.EvalConfig(**PROTOCOL)
    if (cfg.board_size, cfg.max_moves, cfg.mcts_sims) != (24, 280, M.INCUMBENT_SIMS):
        raise Refused("part (b) is not 24x24 / 280 plies / 400 simulations")
    rd = ER.readout_from_eval_config(cfg)

    probe, s, changed = BoardFake(), TwixtState(active_size=24, to_move="red", max_plies_limit=280), 0
    for i in range(3):                                   # the fake must make averaging change the outputs
        for _ in range(7):
            s = s.apply_move(s.legal_moves()[(i * 131 + s.ply * 37) % len(s.legal_moves())])
        moves = s.legal_moves()
        args = (np.transpose(probe.build_input_tensor(s), (1, 2, 0))[None],
                np.array([[r for r, _ in moves]], dtype=np.int32), np.array([[c for _, c in moves]], dtype=np.int32),
                np.ones((1, len(moves)), dtype=np.float32), 24)
        (p, v), (q, w) = probe.infer(*args), MirrorAveragingEvaluator(probe).infer(*args)
        changed += bool(not np.array_equal(p, q) and not np.array_equal(v, w))
    if changed != 3:
        raise Refused(f"the fake's outputs are symmetric on {3 - changed} of 3 positions: averaging would be a no-op")

    log = []

    class Recording(RSH.RootSequentialHalvingMCTS):
        def search_with_root(self, root_state, add_noise=True):
            counts, value, root = super().search_with_root(root_state, add_noise=add_noise)
            log.append({"ply": root_state.ply, "to_move": root_state.to_move,
                        "last_choice": None if self.last_choice is None else list(self.last_choice),
                        "readout": list(M.readout_choice(counts, root, root_state.ply, rd, random.Random(seed)))})
            return counts, value, root

    a = ER.AgentSpec(M.CANDIDATE, "fake-candidate", rd, mcts_sims=M.CANDIDATE_SIMS)
    b = ER.AgentSpec(M.INCUMBENT, "fake-incumbent", rd, mcts_sims=M.INCUMBENT_SIMS)
    classes = {M.CANDIDATE: Recording, M.INCUMBENT: MCTS}
    games = []
    for task in ER.build_agent_pairing_tasks("loop-check-b", a, b, 2, base_seed=seed):
        del log[:]
        incumbent_choices = []
        colour = {True: task.red_agent.agent_id, False: task.black_agent.agent_id}

        def on_search(is_red, _secs, halving, _c=colour):
            if _c[is_red] == M.INCUMBENT and halving:
                incumbent_choices.append(True)

        ev = {M.CANDIDATE: MirrorAveragingEvaluator(BoardFake()), M.INCUMBENT: BoardFake()}
        t = time.monotonic()
        r = M.play_and_build(task, ev[task.red_agent.agent_id], ev[task.black_agent.agent_id], cfg, True,
                             str(out / "b"), classes, on_search)
        moves = {m["ply"]: [m["row"], m["col"]] for m in json.loads(pathlib.Path(r.replay_path).read_text())["moves"]}
        late = [x for x in log if x["ply"] >= 20]
        games.append({
            "game_idx": task.game_idx, "seed": task.seed, "candidate_colour": "red" if task.game_idx % 2 == 0 else
            "black", "reason": r.reason, "winner_agent_id": r.winner_agent_id, "n_moves": r.n_moves,
            "seconds": round(time.monotonic() - t, 3), "candidate_searches": len(log),
            "candidate_searches_from_ply_20": len(late),
            "every_late_search_supplied_last_choice": all(x["last_choice"] is not None for x in late),
            "every_late_move_played_is_last_choice": all(moves[x["ply"]] == x["last_choice"] for x in late),
            "no_early_last_choice": all(x["last_choice"] is None for x in log if x["ply"] < 20),
            "incumbent_supplied_a_choice": bool(incumbent_choices),
            "late_plies_where_the_readout_would_differ": [x["ply"] for x in late if x["readout"] != x["last_choice"]]})
    ok = all(g["candidate_searches_from_ply_20"] > 0 and g["every_late_search_supplied_last_choice"]
             and g["every_late_move_played_is_last_choice"] and g["no_early_last_choice"]
             and not g["incumbent_supplied_a_choice"] for g in games)
    return {"averaging_changed_outputs_on": f"{changed} of 3 probe positions", "games": games,
            "answer": "PASS" if ok else "FAIL"}


def main(rec, out):
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero import mirror_halving_combination as M
    from scripts.GPU.alphazero import root_sequential_halving as RSH
    from scripts.GPU.alphazero.external_learning_stage3 import PROTOCOL
    from scripts.GPU.alphazero.mcts import MCTS
    from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
    from tests.eval_fakes import FakeEvaluator

    M.check_variant(RSH)
    ours = {n: m.__file__ for n, m in list(sys.modules.items())
            if (n == "__main__" or n.split(".")[0] in ("scripts", "tests")) and getattr(m, "__file__", None)}
    rel = sorted(str(pathlib.Path(f).resolve().relative_to(ROOT)) for f in ours.values())
    if git("status", "--porcelain", "--untracked-files=all", "--", *rel):
        raise Refused("a loaded repository module is uncommitted or differs from HEAD")
    rec["modules"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in rel}
    seed = REF.TEST_ONLY_SEED_INTERVALS[0][0]
    if not (REF.seed_is_test_only(seed) and REF.seed_is_test_only(seed + 1)):
        raise Refused("the check's seeds are not test-only")
    rec["seeds"] = [seed, seed + 1]
    t = time.monotonic()
    rec["a"] = part_a(M, ER, PROTOCOL, FakeEvaluator, seed, out)
    rec["a"]["seconds"] = round(time.monotonic() - t, 3)
    t = time.monotonic()
    rec["b"] = part_b(M, ER, RSH, MCTS, MirrorAveragingEvaluator, PROTOCOL, seed, out)
    rec["b"]["seconds"] = round(time.monotonic() - t, 3)
    rec["answer"] = "PASS" if rec["a"]["answer"] == rec["b"]["answer"] == "PASS" else "FAIL"
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "mirror-halving-loop-check/1",
           "script": "docs/superpowers/evidence/mirror-halving-loop-check/01_loop_check.py",
           "head": git("rev-parse", "HEAD")}
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
        b = rec["b"]["games"]
        print(f"COMPLETED: {rec['answer']} (a {rec['a']['answer']}, b {rec['b']['answer']}); "
              + "; ".join(f"b game {g['game_idx']} {g['reason']} {g['n_moves']} plies, {g['candidate_searches_from_ply_20']}"
                          f" halving searches, readout would differ at {len(g['late_plies_where_the_readout_would_differ'])}"
                          for g in b) + f"; {rec['seconds']} s")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
