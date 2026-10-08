"""TACTICAL CANDIDATE -- THE CLOSING SEED ACCOUNTING of the pilot block, DERIVED from committed evidence.

Card `docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §18 A8.2: exposure and retirement
follow actual use, in ONE later, separately authorized closing edit derived from the pilot's committed game
starts. The evaluation closed at the pilot report's STOP (f = 0). This script is READ-ONLY and OUTCOME-BLIND;
of every input it reads only what is named here:
  * the journal (log.jsonl): each line's step name and, for a `game_start` only, its pair_id and arm;
  * each game file -- the one record.json hashed -- through `seed_fields`: its pair_id, arm and seeds
    (red, black). No other field of a game is read;
  * record.json: its status, plan and games' (pair_id, arm, file, sha256); watchdog.json: its two statuses;
  * the committed pilot plan (pinned by sha256) and the registry's four intervals, by parsing.
THE RECONCILIATION: 32 game starts, none repeated, exactly the plan's pairs x arms in order, and the record's
games; every game file's seeds equal its plan pair's; both arms of each pair carry the SAME colour seeds (each
colour seed reused across the two scheduled arms); 32 unique seeds, exactly the block; the block ACCOUNTED
exactly as itself and in neither EXPOSED nor RETIRED; the run COMPLETE; every input committed and unmodified.
RECONCILES (exit 0): every seed of the block started, so the block is EXPOSED WHOLE and RETIRED WHOLE.
Otherwise DOES NOT RECONCILE (1) -- no registry edit. Refused before anything is written (4). One run;
create-only record 02_derive_usage_run.txt.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B <this file>
"""
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = "docs/superpowers/evidence/2026-10-08-tactical-candidate-seed-closing"
OUT = f"{HERE}/02_derive_usage_run.txt"
PILOT = "docs/superpowers/evidence/tactical-candidate-pilot"
PLAN = "docs/superpowers/evidence/tactical-candidate-plans/pilot-s0.json"
PLAN_SHA256 = "b4e9e00a3bb9ca9b3ee7de8c9a4a8f1a55b04e168cd82debb2d9f67a92e70b61"
REGISTRY = "scripts/GPU/alphazero/e4_screen_reference.py"
ARMS = ("A", "B")
COLOURS = ("red", "black")
NAMES = ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS",
         "TEST_ONLY_SEED_INTERVALS")


class Refused(Exception):
    """A precondition does not hold: nothing is derived or written."""


_LINES = []


def say(line=""):
    _LINES.append(line)
    print(line, flush=True)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


# ─────────────────────────────── the whitelisted reads ───────────────────────────────

def seed_fields(game):
    """The ONLY fields of a game the derivation reads."""
    seeds = game["seeds"]
    return {"pair_id": game["pair_id"], "arm": game["arm"], "seeds": {c: seeds[c] for c in COLOURS}}


def game_starts(lines):
    """[(pair_id, arm)] of every `game_start`, in journal order; nothing else of any line is read."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["step"] == "game_start":
            obs = entry["observation"]
            out.append((obs["pair_id"], obs["arm"]))
    return out


def registry_intervals(src):
    """{name: tuple of intervals} for the four registries, by parsing; each assigned exactly once."""
    found = {n: [] for n in NAMES}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in found:
            found[node.targets[0].id].append(ast.literal_eval(node.value))
    if any(len(v) != 1 for v in found.values()):
        raise Refused("a registry is not assigned exactly once")
    return {n: tuple(tuple(iv) for iv in v[0]) for n, v in found.items()}


# ─────────────────────────────── the reconciliation, pure ───────────────────────────────

def reconcile(plan, starts, record_games, seeds_by_game, registries, statuses):
    """Faults, and the block (or None). `seeds_by_game`: [seed_fields(game)] in record order."""
    faults = []
    want = [(p["pair_id"], arm) for p in plan["pairs"] for arm in ARMS]
    if len(starts) != len(set(starts)):
        faults.append("a game started more than once")
    if starts != want:
        faults.append(f"the {len(starts)} game starts are not the plan's {len(want)} pairs x arms, in order")
    if [(g["pair_id"], g["arm"]) for g in record_games] != starts:
        faults.append("the record's games are not the started games")
    if [(s["pair_id"], s["arm"]) for s in seeds_by_game] != [(g["pair_id"], g["arm"]) for g in record_games]:
        faults.append("the game files are not the record's games")
    planned = {p["pair_id"]: {c: p["seeds"][c] for c in COLOURS} for p in plan["pairs"]}
    for s in seeds_by_game:
        if s["seeds"] != planned.get(s["pair_id"]):
            faults.append(f"game {s['pair_id']}-{s['arm']}: its seeds are not its plan pair's")
    by_pair = {}
    for s in seeds_by_game:
        by_pair.setdefault(s["pair_id"], []).append(s["seeds"])
    for pid, both in sorted(by_pair.items()):
        if len(both) != len(ARMS) or any(x != both[0] for x in both):
            faults.append(f"pair {pid}: its colour seeds are not reused across both arms")
    started = [planned[pid][c] for pid, _arm in starts if pid in planned for c in COLOURS]
    seeds = sorted(set(started))
    block = (seeds[0], seeds[-1] + 1) if seeds else None
    if not seeds or seeds != list(range(*block)) or len(seeds) != 2 * len(plan["pairs"]):
        faults.append(f"the started seeds ({len(seeds)} unique) are not one contiguous block of "
                      f"{2 * len(plan['pairs'])}")
    if block is not None:
        if registries["ACCOUNTED_SEED_INTERVALS"].count(block) != 1:
            faults.append("the block is not ACCOUNTED exactly as itself")
        for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
            if any(lo < block[1] and block[0] < hi for lo, hi in registries[name]):
                faults.append(f"seeds of the block are already in {name}")
    if statuses != ("COMPLETE", "COMPLETE", "COMPLETE"):
        faults.append(f"the run is not COMPLETE: {statuses}")
    return faults, block


# ─────────────────────────────── the controls ───────────────────────────────

def _world(n_pairs=3, lo=500):
    pairs = [{"pair_id": f"p-{k}", "seeds": {"red": lo + 2 * k, "black": lo + 2 * k + 1}} for k in range(n_pairs)]
    plan = {"pairs": pairs}
    starts = [(p["pair_id"], arm) for p in pairs for arm in ARMS]
    games = [{"pair_id": pid, "arm": arm} for pid, arm in starts]
    seeds = [{"pair_id": pid, "arm": arm, "seeds": dict(next(p["seeds"] for p in pairs if p["pair_id"] == pid))}
             for pid, arm in starts]
    regs = {"ACCOUNTED_SEED_INTERVALS": ((100, 132), (lo, lo + 2 * n_pairs)), "EXPOSED_SEED_INTERVALS": ((100, 132),),
            "RETIRED_SEED_INTERVALS": ((100, 132),), "TEST_ONLY_SEED_INTERVALS": ()}
    return plan, starts, games, seeds, regs, ("COMPLETE",) * 3


def _controls():
    def ok(edit):
        w = list(_world())
        edit(w)
        return not reconcile(*w)[0]

    def setw(i, v):
        return lambda w: w.__setitem__(i, v)
    plan, starts, games, seeds, regs, _st = _world()
    swapped = [dict(s, seeds=dict(s["seeds"])) for s in seeds]
    swapped[1]["seeds"]["red"] = seeds[2]["seeds"]["red"]          # arm B of pair 0 carries pair 1's red
    both_wrong = [dict(s, seeds=dict(s["seeds"])) for s in seeds]
    for s in both_wrong[:2]:
        s["seeds"]["black"] = 999                                    # BOTH arms agree, but not the plan
    gap_plan = {"pairs": [dict(p, seeds=dict(p["seeds"])) for p in plan["pairs"]]}
    gap_plan["pairs"][2]["seeds"]["black"] = 507                     # one seed outside a contiguous block
    gap_seeds = [dict(s, seeds=dict(next(p["seeds"] for p in gap_plan["pairs"] if p["pair_id"] == s["pair_id"])))
                 for s in seeds]
    cases = [
        ("a sound world reconciles", lambda w: None, True),
        ("a game start missing", setw(1, starts[:-1]), False),
        ("a game started twice", setw(1, starts + starts[-1:]), False),
        ("the starts out of plan order", setw(1, [starts[1], starts[0]] + starts[2:]), False),
        ("the record's games are not the started games", setw(2, games[:-1]), False),
        ("the game files are not the record's games", setw(3, seeds[::-1]), False),
        ("one arm's seeds differ from the other arm's", setw(3, swapped), False),
        ("both arms agree but not with the plan", setw(3, both_wrong), False),
        ("the started seeds are not one contiguous block",
         lambda w: (w.__setitem__(0, gap_plan), w.__setitem__(3, gap_seeds)), False),
        ("the block is not ACCOUNTED", setw(4, dict(regs, ACCOUNTED_SEED_INTERVALS=((100, 132),))), False),
        ("the block ACCOUNTED twice", setw(4, dict(regs, ACCOUNTED_SEED_INTERVALS=regs["ACCOUNTED_SEED_INTERVALS"] * 2)),
         False),
        ("a seed of the block already EXPOSED", setw(4, dict(regs, EXPOSED_SEED_INTERVALS=((505, 506),))), False),
        ("a seed of the block already RETIRED", setw(4, dict(regs, RETIRED_SEED_INTERVALS=((499, 501),))), False),
        ("an adjacent EXPOSED interval is no overlap", setw(4, dict(regs, EXPOSED_SEED_INTERVALS=((490, 500),))), True),
        ("the run not COMPLETE", setw(5, ("COMPLETE", "COMPLETE", "STOP")), False),
    ]
    out = [(label, ok(edit) is want, f"expected {want}") for label, edit, want in cases]
    try:
        seed_fields({"pair_id": "p", "arm": "A", "seeds": {"red": 1}})
        out.append(("a game without both colour seeds is refused", False, ""))
    except KeyError:
        out.append(("a game without both colour seeds is refused", True, ""))
    g = seed_fields({"pair_id": "p", "arm": "A", "seeds": {"red": 1, "black": 2, "x": 3}, "moves": [[0, 0]]})
    out.append(("seed_fields returns ONLY pair_id, arm and the two colour seeds",
                g == {"pair_id": "p", "arm": "A", "seeds": {"red": 1, "black": 2}}, ""))
    lines = [json.dumps({"step": "game_start", "observation": {"pair_id": "p", "arm": "A"}}),
             json.dumps({"step": "game", "observation": {"anything": 1}}), ""]
    out.append(("game_starts reads only game_start entries", game_starts(lines) == [("p", "A")], ""))
    return out


# ─────────────────────────────── the run ───────────────────────────────

def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _inputs():
    """Every input, read as named above; refuses unless all are committed, unmodified and pinned."""
    if os.path.lexists(ROOT / OUT):
        raise Refused(f"{OUT} exists: the derivation runs once (create-only)")
    if _git("status", "--porcelain").stdout.strip():
        raise Refused("the working tree is not clean: the derivation reads the COMMITTED state")
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    d = ROOT / PILOT
    record_raw = (d / "record.json").read_bytes()
    record = json.loads(record_raw)
    games = [{k: g[k] for k in ("pair_id", "arm", "file", "sha256")} for g in record["games"]]
    paths = [ROOT / PLAN, ROOT / REGISTRY] + [d / n for n in ("record.json", "watchdog.json", "log.jsonl")] + \
        [d / "games" / g["file"] for g in games]
    try:
        F.check_durable([str(p) for p in paths], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise Refused(f"an input is not committed and unmodified: {e}") from None
    plan_raw = (ROOT / PLAN).read_bytes()
    if _sha(plan_raw) != PLAN_SHA256:
        raise Refused("the plan is not the pilot's committed plan")
    plan = json.loads(plan_raw)
    if record["plan"] != plan:
        raise Refused("the pilot record ran another plan")
    seeds = []
    for g in games:
        raw = (d / "games" / g["file"]).read_bytes()
        if _sha(raw) != g["sha256"]:
            raise Refused(f"{g['file']} is not the file the record hashed")
        seeds.append(seed_fields(json.loads(raw)))
    watchdog = json.loads((d / "watchdog.json").read_text(encoding="utf-8"))
    return {"head": _git("rev-parse", "HEAD").stdout.strip(), "plan": plan,
            "starts": game_starts((d / "log.jsonl").read_text(encoding="utf-8").splitlines()),
            "games": games, "seeds": seeds,
            "registries": registry_intervals((ROOT / REGISTRY).read_text(encoding="utf-8")),
            "statuses": (watchdog["status"], watchdog["record_status"], record["status"]),
            "files": len(paths)}


def _write(out):
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)       # create-only
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main(argv):
    if len(argv) > 1:
        raise Refused("this script takes no arguments")
    x = _inputs()
    say("TACTICAL CANDIDATE -- CLOSING SEED ACCOUNTING OF THE PILOT BLOCK (card §18 A8.2; read-only, outcome-blind)")
    say("=" * 96)
    say(f"HEAD {x['head']}; working tree clean; {x['files']} inputs committed and unmodified "
        f"(plan sha256 {PLAN_SHA256}, registry, pilot record, watchdog, journal, {len(x['games'])} game files)")
    faults, block = reconcile(x["plan"], x["starts"], x["games"], x["seeds"], x["registries"], x["statuses"])
    seeds = {c: sorted({s["seeds"][c] for s in x["seeds"]}) for c in COLOURS}
    say(f"run status (watchdog, its record status, record): {x['statuses']}")
    say(f"game starts in the journal: {len(x['starts'])}, distinct {len(set(x['starts']))}; the plan's pairs x arms: "
        f"{2 * len(x['plan']['pairs'])}; the record's games: {len(x['games'])}")
    say(f"seed fields read from {len(x['seeds'])} game files: {len(seeds['red'])} distinct red, "
        f"{len(seeds['black'])} distinct black, {len(set(seeds['red']) | set(seeds['black']))} distinct in all; "
        f"each pair's two arms carry the same colour seeds: "
        f"{not any('reused' in f for f in faults)}")
    say(f"the block the started seeds make: {block}; ACCOUNTED holds it as itself: "
        f"{block is not None and x['registries']['ACCOUNTED_SEED_INTERVALS'].count(block) == 1}; "
        f"in EXPOSED: {any('EXPOSED' in f for f in faults)}; in RETIRED: {any('RETIRED' in f for f in faults)}")
    say(f"faults: {faults if faults else 'none'}")
    say("\nCONTROLS (synthetic)")
    say("-" * 96)
    results = _controls()
    for label, ok, detail in results:
        say(f"  {'PASS' if ok else 'FAIL':6s} {label}{'  [' + detail + ']' if detail and not ok else ''}")
    passed = sum(ok for _l, ok, _d in results)
    say(f"\ncontrols passed: {passed}/{len(results)}")
    good = not faults and passed == len(results) > 0
    say("VERDICT: " + (f"RECONCILES -- all {block[1] - block[0]} seeds of {block} started, each colour seed in both "
                       "arms of its pair: the pilot block is EXPOSED WHOLE and RETIRED WHOLE" if good else
                       "DOES NOT RECONCILE -- no registry edit"))
    _write(ROOT / OUT)
    return 0 if good else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
