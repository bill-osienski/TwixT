"""SELF-PLAY DIAGNOSTIC -- the read-only one-ply scan of the committed replays. No model, no search, no seed.

Every pre-move position of every replay in games/ goes through h5_weakness_review.scan_trajectory, the H5 review's
exact one-ply labels (card docs/superpowers/2026-10-01-t1j-h5-weakness-review-card.md §7 C1):
  MISSED_WIN      the mover had an immediate win and played something else
  MISSED_DEFENCE  no immediate win; the opponent threatened one; some move left none; the move played left one
  UNDEFENDABLE    as above, but no move left none (not an error)
Only MISSED_WIN and MISSED_DEFENCE are verifiable mistakes, with one exclusion: the final move of a game that
ended at the 280-ply cap is never MISSED_DEFENCE (the opponent has no next turn); it is recorded as
CAP_FINAL_EXCLUDED. For each mistake: the game, ply, mover, the board, the move played, the better moves (W, or
the saving moves), whether the opponent won on the next ply, and the SEARCH'S PREFERRED move (the most-visited
root child, from the replay) apart from the SAMPLED move played -- the readout samples at temperature 1.0 for
plies 0-19 and 0.1 after, so a wrong played move does not by itself identify a search defect -- with whether
each is among the better moves, the played move's visit rank and count, the root value, and the two
most-visited moves. Anything less certain is not labelled here.

Runs only on a clean complete run: outer.json outer exit 0 with no survivors, record.json COMPLETED, exactly the
4 replays; every input committed and unmodified. Reads only this run's games/. Writes one create-only record,
04_scan.json, and prints a summary. Exit 0 written / 4 refused.

    python -B <this file>
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEMPERATURE_PLIES = 20                                   # Stage B PROTOCOL opening_temp_plies


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def main():
    out = HERE / "04_scan.json"
    if out.exists():
        print("REFUSED: 04_scan.json exists (create-only)", file=sys.stderr)
        return 4
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import h5_weakness_review as RV

    outer = json.loads((HERE / "outer.json").read_text()) if (HERE / "outer.json").exists() else {}
    verdict = json.loads((HERE / "record.json").read_text()).get("verdict") if (HERE / "record.json").exists() else None
    paths = sorted((HERE / "games").glob("game_*.json"))
    if outer.get("outer_exit") != 0 or outer.get("survivors", "x").strip() or verdict != "COMPLETED" or len(paths) != 4:
        print("REFUSED: not a clean complete run (outer exit 0, no survivors, COMPLETED, 4 replays)", file=sys.stderr)
        return 4
    rels = [str(p.relative_to(ROOT)) for p in paths] + [str((HERE / n).relative_to(ROOT)) for n in ("outer.json",
                                                                                                     "record.json")]
    if any(git("ls-files", "--error-unmatch", "--", r).returncode or
                       git("diff", "--quiet", "HEAD", "--", r).returncode for r in rels):
        print("REFUSED: no replays, or a replay is uncommitted or modified", file=sys.stderr)
        return 4
    rec = {"kind": "self-play-diagnostic-scan/1", "head": git("rev-parse", "HEAD").stdout.strip(),
           "labels": "h5_weakness_review.scan_trajectory (card §7 C1)", "games": [], "mistakes": []}
    rec["excluded"] = []
    for p, rel in zip(paths, rels):
        g = json.loads(p.read_text())
        plies = g["moves"]
        moves = [(m["row"], m["col"]) for m in plies]
        scan = RV.scan_trajectory(moves, ["incumbent"] * len(moves), "cap" if g["reason"] == "state_cap" else "win")
        counts = {lab: sum(s["label"] == lab for s in scan) for lab in ("MISSED_WIN", "MISSED_DEFENCE", "UNDEFENDABLE",
                                                                       "DEFENDED")}
        rec["games"].append({"replay": rel, "seed": g["seed"], "winner": g["winner"], "reason": g["reason"],
                             "n_moves": g["n_moves"], "positions": len(scan), "labels": counts})
        for s in scan:
            if s["label"] == "MISSED_DEFENCE" and s["reply_available"] is False:
                counts["MISSED_DEFENCE"] -= 1
                counts["CAP_FINAL_EXCLUDED"] = 1
                rec["excluded"].append({"replay": rel, "ply_index": s["ply"] - 1, "label": "CAP_FINAL_EXCLUDED"})
                continue
            if s["label"] not in ("MISSED_WIN", "MISSED_DEFENCE"):
                continue
            i = s["ply"] - 1
            r = plies[i]
            better = s["W"] if s["label"] == "MISSED_WIN" else s["saving_moves"]
            pref = [r["top2"][0]["row"], r["top2"][0]["col"]] if r["top2"] else None
            rec["mistakes"].append({
                "replay": rel, "ply_index": i, "mover": s["mover"], "label": s["label"], "played": s["move"],
                "better": better, "search_preferred": pref,
                "played_is_search_preferred": pref == s["move"],
                "search_preferred_is_better": pref in better,
                "readout_temperature": 1.0 if i < TEMPERATURE_PLIES else 0.1,
                "opponent_threats": s["T"], "opponent_won_next_ply": s["opponent_won_next_ply"],
                "root": {k: r[k] for k in ("root_value", "root_top1_share", "selected_visit_rank",
                                           "selected_visit_count", "root_total_visits", "top2",
                                           "readout_overrode_leader")},
                "board_before": RV.render(moves, upto=i)})
    with open(out, "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    for g in rec["games"]:
        print(f"{g['replay']}: {g['n_moves']} plies, {g['reason']}, winner {g['winner']}; {g['labels']}")
    print(f"verifiable mistakes: {len(rec['mistakes'])}")
    if not rec["mistakes"]:
        print("No missed immediate win or avoidable one-move loss found in these four games.")
    for m in rec["mistakes"]:
        print(f"  {m['label']} {m['replay']} ply {m['ply_index']} {m['mover']} played {m['played']}, better "
              f"{m['better'][:8]}{'...' if len(m['better']) > 8 else ''}; search preferred {m['search_preferred']} "
              f"(better: {m['search_preferred_is_better']}); readout temperature {m['readout_temperature']}; "
              f"opponent won next ply {m['opponent_won_next_ply']}")
    print(f"excluded (capped final move): {len(rec['excluded'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
