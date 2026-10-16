"""SELF-PLAY DIAGNOSTIC -- ONE INSTRUMENTED SEARCH of the missed-defence position (game 3, ply index 41, black to
move; 04_scan.json). A FRESH COLD SEARCH: the root is rebuilt from the committed replay and searched once with a
test-only seed, so it cannot reproduce the game's advanced RNG stream exactly.

* the position: the replay's first 41 moves applied to a fresh 24x24 state (280-ply cap); refused unless black is
  to move, red's immediate wins are exactly 04_scan.json's threats and (21,6) is exactly the saving set (the H5
  one-ply functions, re-run here on this position only);
* one load of calib020_0001 through eval_runner's factory (evaluation mode checked); one MCTS from cfg_from(Stage B
  PROTOCOL) -- 400 simulations, c_puct 1.5, absolute FPU 0.0, no root noise -- through search_with_root, the
  sequential path the game used; RNG random.Random(TEST_ONLY_SEED_INTERVALS[0][0] ^ black's search mask);
* instrumentation, read-only: the instance's _select_child is wrapped; at every ROOT selection, before the
  original runs (it alone draws the tie-break), the PUCT score of every root move is recomputed with mcts.py's own
  formula -- the best VISITED move (score, Q, U, visits) and the best UNEXPLORED move (score, prior), the number
  tied at the top, and the move the original then chose. A recomputed best score that differs from the chosen
  move's score is a STOP (the recomputation would not be the code's).
Recorded after the search: every root move's prior, prior rank, visits and Q (root perspective, None if
unvisited); (21,6)'s prior, rank, visits, Q, network value, and every opponent reply searched beneath it.
Answers, descriptively: was (21,6) never reached, reached and evaluated pessimistically, or reached and examined
further? No parameter change, no game, no training.

  --checkpoint PATH (absolute)   --out DIR (exists, without record.json)
Exit 0 COMPLETED / 2 STOP / 3 STOP_ERROR / 4 REFUSED. Run under `timeout -k 10 50` (60 s in all).
"""
import hashlib
import json
import math
import pathlib
import random
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
REPLAY, PLY, SAVE = HERE / "games/game_000003.json", 41, (21, 6)
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


class Refused(Exception):
    pass


class Stop(Exception):
    pass


def position(RV, D0, TwixtState):
    g = json.loads(REPLAY.read_text())
    scan = json.loads((HERE / "04_scan.json").read_text())["mistakes"]
    [m] = [m for m in scan if m["replay"].endswith(REPLAY.name) and m["ply_index"] == PLY]
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for mv in g["moves"][:PLY]:
        s = s.apply_move((mv["row"], mv["col"]))
    T = D0.one_ply_threats(D0._flip(s))
    saving = RV.saving_moves_of(s, T)
    if s.to_move != "black" or D0.one_ply_threats(s) or [list(t) for t in T] != m["opponent_threats"] or \
            saving != [SAVE] or [g["moves"][PLY]["row"], g["moves"][PLY]["col"]] != m["played"]:
        raise Refused("the rebuilt position is not 04_scan.json's missed defence")
    return s, {"ply_index": PLY, "to_move": s.to_move, "red_threats": [list(t) for t in T], "saving": [list(SAVE)],
               "played_in_game": m["played"]}


def main(rec, argv):
    ckpt = pathlib.Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
    if ckpt is None or not ckpt.is_absolute() or not ckpt.is_file():
        raise Refused("--checkpoint must name an existing file by absolute path")
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import d0_postmortem as D0
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import eval_runner as ER
    from scripts.GPU.alphazero import h5_weakness_review as RV
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, decode_move, encode_move
    from scripts.GPU.alphazero.search_distillation_stage_b import PROTOCOL

    state, rec["position"] = position(RV, D0, TwixtState)
    seed = REF.TEST_ONLY_SEED_INTERVALS[0][0]
    if not REF.seed_is_test_only(seed):
        raise Refused("the diagnostic seed is not test-only")
    if hashlib.sha1(ckpt.read_bytes()).hexdigest() != SHA1:
        raise Refused("the checkpoint is not the pinned calib020_0001")
    ev = ER._default_evaluator_factory(str(ckpt))
    mods = ev.network.named_modules()
    rec["modules_in_training_mode"] = sum(bool(x.training) for _, x in mods)
    if ev.network.training or rec["modules_in_training_mode"]:
        raise Stop("the network is not in evaluation mode")
    cfg = ER.cfg_from(ER.EvalConfig(**PROTOCOL))
    if (cfg.n_simulations, cfg.c_puct, cfg.fpu_value, cfg.fpu_policy_mass_reduction) != (400, 1.5, 0.0, None):
        raise Refused("the search settings are not 400 sims, c_puct 1.5, absolute FPU 0.0")
    rec["search"] = {"simulations": cfg.n_simulations, "c_puct": cfg.c_puct, "fpu_value": cfg.fpu_value,
                     "add_noise": False, "seed": "TEST_ONLY_SEED_INTERVALS[0][0] ^ 0x5A5A5A (black's search mask)",
                     "cold": "fresh tree and RNG; not the game's advanced RNG stream"}
    mcts = MCTS(ev, cfg, random.Random(seed ^ 0x5A5A5A))
    original, root_box, log = mcts._select_child, {}, []

    def mv(move_id):
        return list(decode_move(move_id))

    def wrapped(node, pending_node_ids=None):
        if node is not root_box.get("root"):
            return original(node, pending_node_ids)
        sqrt_parent = math.sqrt(node.visit_count + 1)
        visited, unexplored = None, None
        scores = {}
        for move_id, prior in node.priors.items():
            child = node.children.get(move_id)
            n = child.visit_count if child is not None else 0
            q = -child.q_value if n > 0 else cfg.fpu_value
            u = cfg.c_puct * prior * sqrt_parent / (1 + n)
            scores[move_id] = q + u
            row = (q + u, move_id, q, u, n, prior)
            if n > 0:
                visited = row if visited is None or row[0] > visited[0] else visited
            else:
                unexplored = row if unexplored is None or row[0] > unexplored[0] else unexplored
        best = max(scores.values())
        move_id, child = original(node, pending_node_ids)
        if abs(scores[move_id] - best) > 1e-8:
            raise Stop(f"recomputed best {best} is not the chosen move's score {scores[move_id]}")
        save = scores.get(encode_move(*SAVE))
        log.append({"root_visits": node.visit_count, "chosen": mv(move_id),
                    "chosen_was_unexplored": child is None or child.visit_count == 0,
                    "tied_at_top": sum(abs(x - best) <= 1e-8 for x in scores.values()),
                    "best_visited": None if visited is None else {"move": mv(visited[1]), "score": visited[0],
                                                                  "q": visited[2], "u": visited[3],
                                                                  "visits": visited[4]},
                    "best_unexplored": None if unexplored is None else {"move": mv(unexplored[1]),
                                                                        "score": unexplored[0],
                                                                        "prior": unexplored[5]},
                    "saving_move_score": save})
        return move_id, child

    # search_with_root builds its own root; capture it at its first selection (the root is the first node selected
    # from, and it is the only node with no parent).
    first = {"seen": False}

    def capture(node, pending_node_ids=None):
        if not first["seen"] and node.parent is None:
            first["seen"], root_box["root"] = True, node
        return wrapped(node, pending_node_ids)

    mcts._select_child = capture
    t = time.monotonic()
    counts, value, root = mcts.search_with_root(state, add_noise=False)
    rec["search_seconds"] = round(time.monotonic() - t, 3)
    if root is not root_box.get("root") or len(log) != cfg.n_simulations:
        raise Stop(f"instrumentation saw {len(log)} root selections, not {cfg.n_simulations}")

    ranked = sorted(root.priors, key=lambda k: (-root.priors[k], decode_move(k)))
    rank = {k: i + 1 for i, k in enumerate(ranked)}
    moves = []
    for k in ranked:
        c = root.children.get(k)
        n = c.visit_count if c is not None else 0
        moves.append({"move": mv(k), "prior": root.priors[k], "prior_rank": rank[k], "visits": n,
                      "q_root": -c.q_value if n else None})
    sid = encode_move(*SAVE)
    sc = root.children.get(sid)
    replies = []
    if sc is not None:
        replies = [{"move": mv(k), "visits": c.visit_count, "q_red": c.q_value if c.visit_count else None}
                   for k, c in sorted(sc.children.items()) if c.visit_count > 0]
    sn = sc.visit_count if sc is not None else 0
    visited_q = [m["q_root"] for m in moves if m["visits"]]
    rec["root"] = {"value": value, "visits": root.visit_count, "legal": len(counts), "children_visited": len(visited_q),
                   "max_visits": max(m["visits"] for m in moves), "nn_value": root.nn_value,
                   "visited_q_root": {"min": min(visited_q), "max": max(visited_q),
                                      "mean": sum(visited_q) / len(visited_q)},
                   "moves": moves}
    rec["saving_move"] = {"move": list(SAVE), "prior": root.priors[sid], "prior_rank": rank[sid], "visits": sn,
                          "q_root": -sc.q_value if sn else None,
                          "network_value_red_to_move": sc.nn_value if sc is not None else None,
                          "first_selected_at_root_visit": next((e["root_visits"] for e in log
                                                                if e["chosen"] == list(SAVE)), None),
                          "visited_q_rank": (1 + sum(q > -sc.q_value for q in visited_q)) if sn else None,
                          "opponent_replies_searched": replies}
    rec["root_selections"] = log
    rec["answer"] = ("NEVER_REACHED" if sn == 0 else "REACHED_AND_EXAMINED_FURTHER" if replies else
                     "REACHED_ONCE_NOT_EXAMINED_FURTHER")
    return "COMPLETED"


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        sys.exit(EXIT["REFUSED"])
    rec = {"kind": "self-play-diagnostic-probe/1", "script": "docs/superpowers/evidence/self-play-diagnostic/05_probe.py",
           "head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    try:
        verdict = main(rec, sys.argv)
    except Refused as e:
        verdict, rec["reason"] = "REFUSED", str(e)
    except Stop as e:
        verdict, rec["reason"] = "STOP", str(e)
    except Exception:  # noqa: BLE001 -- the record boundary covers every Python-level failure
        verdict, rec["reason"] = "STOP_ERROR", traceback.format_exc()
    rec["verdict"] = verdict
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if verdict == "COMPLETED":
        s, r = rec["saving_move"], rec["root"]
        print(f"COMPLETED: {rec['answer']}; (21,6) prior {s['prior']:.4g} rank {s['prior_rank']}, visits "
              f"{s['visits']}, Q {s['q_root']}; root {r['visits']} visits over {r['children_visited']} children, "
              f"max {r['max_visits']}, value {r['value']:.4f}")
    else:
        print(f"{verdict}: {rec.get('reason')}", file=sys.stderr)
    sys.exit(EXIT[verdict])
