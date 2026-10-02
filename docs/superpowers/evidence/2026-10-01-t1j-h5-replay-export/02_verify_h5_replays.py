"""H5 REPLAY EXPORT -- VERIFY every file against its source, then pick the viewer set.

Independent of the exporter's own checks: the committed results are parsed HERE,
record by record, and each of the 592 files is compared with them -- id and task,
the move list (turn, player, row, col), n_moves, the transcript digest (the file's
== the result's == recomputed from the plies), identities by colour (against
game_start and the arm's colour assignment), seed, pair, arm, outcome, terminal
reason, starting player, source-results sha256, and the H5 labels. Finally each file
with H4's three provenance strings restored must EQUAL h4_runner.export_game's
output: the relabel is the only difference.

Viewer set: both arms of the first and last pair of every segment (16 files), plus
the first file, in segment and schedule order, of any terminal category -- red win,
black win, cap -- the 16 lack. Written create-only to 03_viewer_selection.json.
"""
import copy
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.GPU.alphazero import h4_runner as R4                          # noqa: E402
from scripts.GPU.alphazero import h5_runner as H5                          # noqa: E402

H4_LABELS = {"config_hash": "h4-recorded-evidence", "mode": "recorded_h4",
             "evidence_note": "derived from the H4 results file; a viewer export, not a result"}
H5_LABELS = {"config_hash": "h5-recorded-evidence", "mode": "recorded_h5",
             "evidence_note": "derived from the H5 results file; a viewer export, not a result"}
INCUMBENT_COLOUR = {"A": "red", "B": "black"}


def raw_games(path):
    order, games = [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = r["record_type"]
            if t == "game_start":
                order.append(r["task_id"])
                games[r["task_id"]] = {"start": r, "plies": [], "result": None}
            elif t == "ply":
                games[r["task_id"]]["plies"].append(r)
            elif t == "game_result":
                games[r["task_id"]]["result"] = r
    return order, games


def category(f):
    return "cap" if f["meta"]["reason"] == "cap" else f"{f['winner']} win"


def checks(f, g, sha, exported):
    s, plies, res = g["start"], g["plies"], g["result"]
    restored = copy.deepcopy(f)
    restored["config_hash"] = H4_LABELS["config_hash"]
    restored["meta"]["mode"] = H4_LABELS["mode"]
    restored["meta"]["evidence_note"] = H4_LABELS["evidence_note"]
    return {
        "id_and_task": f["id"] == s["task_id"] == f["meta"]["task_id"] == res["task_id"],
        "moves": [(m["turn"], m["player"], m["row"], m["col"]) for m in f["moves"]]
                 == [(p["ply"], p["mover"], p["move"][0], p["move"][1]) for p in plies],
        "n_moves": f["meta"]["n_moves"] == len(f["moves"]) == res["plies"] == len(plies),
        "digest": f["meta"]["transcript_digest"] == res["transcript_digest"]
                  == R4.transcript_digest_of(plies, res),
        "identities_by_colour": (f["meta"]["players"] == {s["t1j_colour"]: "T1j (theirs)",
                                                          s["incumbent_colour"]: "Incumbent (ours)"}
                                 and s["incumbent_colour"] == INCUMBENT_COLOUR[s["arm"]]),
        "seed": f["seed"] == s["seed"],
        "pair_and_arm": f["meta"]["pair_id"] == s["pair_id"] == res["pair_id"]
                        and f["meta"]["arm"] == s["arm"] == res["arm"],
        "outcome": f["winner"] == (res["winner"] if res["winner"] is not None else "draw"),
        "terminal_reason": f["meta"]["reason"] == res["terminal_reason"],
        "starting_player": f["starting_player"] == "red",
        "source_sha256": f["meta"]["source_sha256"] == sha,
        "h5_labels": {"config_hash": f["config_hash"], "mode": f["meta"]["mode"],
                      "evidence_note": f["meta"]["evidence_note"]} == H5_LABELS,
        "only_the_labels_differ_from_H4s_exporter":
            restored == json.loads(json.dumps(exported)),
    }


def main():
    failures, total, firsts, selection = [], 0, {}, []
    for k in H5.SEGMENTS:
        results = ROOT / H5.EVIDENCE_DIRS[k] / "results.jsonl"
        sha = hashlib.sha256(results.read_bytes()).hexdigest()
        order, games = raw_games(results)
        replays = results.parent / "replays"
        names = sorted(p.stem for p in replays.glob("*.json"))
        if names != sorted(order) or len(order) != 148:
            failures.append(f"segment {k}: files {len(names)} do not match the {len(order)} games")
            continue
        exported = {g["task_id"]: R4.export_game(g, str(results))
                    for g in R4.load_games(str(results))}
        for tid in order:
            f = json.loads((replays / f"{tid}.json").read_text(encoding="utf-8"))
            bad = [n for n, ok in checks(f, games[tid], sha, exported[tid]).items() if not ok]
            if bad:
                failures.append(f"{tid}: {bad}")
            firsts.setdefault(category(f), f"{H5.EVIDENCE_DIRS[k]}/replays/{tid}.json")
            total += 1
        pairs = sorted({games[t]["start"]["pair_id"] for t in order})
        for pid in (pairs[0], pairs[-1]):
            for arm in ("A", "B"):
                selection.append(f"{H5.EVIDENCE_DIRS[k]}/replays/{pid}-{arm}.json")
        print(f"segment {k}: {len(order)} files, source sha256 {sha[:12]}..., "
              f"{sum(1 for x in failures if x.startswith(f'h5s{k}-'))} failing")
    print(f"files verified: {total}; failures: {len(failures)}")
    for x in failures[:20]:
        print("  FAIL", x)
    print("checks per file: id_and_task, moves, n_moves, digest (file == result == "
          "recomputed), identities_by_colour, seed, pair_and_arm, outcome, terminal_reason, "
          "starting_player, source_sha256, h5_labels, only_the_labels_differ_from_H4s_exporter")
    covered = {category(json.loads((ROOT / p).read_text())) for p in selection}
    added = [firsts[c] for c in ("red win", "black win", "cap") if c in firsts and c not in covered]
    print(f"viewer set: {len(selection)} first/last-pair files cover {sorted(covered)}; "
          f"added for missing categories: {added or 'none'}")
    with open(HERE / "03_viewer_selection.json", "x", encoding="utf-8") as fh:   # create-only
        json.dump(selection + added, fh, indent=1)
    return 0 if not failures and total == 592 else 1


if __name__ == "__main__":
    sys.exit(main())
