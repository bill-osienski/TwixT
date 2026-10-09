"""EXTERNAL-ENGINE LEARNING STAGE 1 -- THE CLOSING SEED ACCOUNTING of the stage-1 block, DERIVED from committed
evidence.

Card `docs/superpowers/2026-10-08-external-engine-learning-viability.md` (seeds: "actual exposure retirement");
the registry comment of the block: exposure follows actual use, in a later closing edit. The stage-1 run is
committed at evidence/external-engine-learning-stage1/. This script is READ-ONLY and OUTCOME-BLIND; of every
input it reads only what is named here:
  * progress.jsonl: each line's `event` and, for a `game_start` only, its game_id and seed;
  * each game file named by the plan, through `seed_fields`: its game_id, family and seed. No other field;
  * record.json: steps.plan.sha256 and steps.openings (the six four-ply openings, played before any game);
  * finalized.json: status, group_reaped, exit_code and verdict; MANIFEST.sha256: every line;
  * the committed plan (pinned by sha256) and the registry's four intervals, by parsing.
THE RECONCILIATION: 12 game starts, none repeated, exactly the plan's games and seeds in order; every game file's
seed fields equal its plan entry's; each family's recorded opening equals the opening its plan seed draws
(recomputed here with the run's own `generate_opening`), so each opening seed was drawn; the used seeds are 18
unique contiguous seeds, exactly the plan's block; the block ACCOUNTED exactly as itself and in neither EXPOSED
nor RETIRED; the run finalized (exited, group reaped); every input committed, unmodified and the file the
manifest hashed.
RECONCILES (exit 0): every seed of the block was used, so the block is EXPOSED WHOLE and RETIRED WHOLE.
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
HERE = "docs/superpowers/evidence/external-engine-learning-seed-closing"
OUT = f"{HERE}/02_derive_usage_run.txt"
RUN = "docs/superpowers/evidence/external-engine-learning-stage1"
PLAN = "docs/superpowers/evidence/external-engine-learning-plans/stage1.json"
PLAN_SHA256 = "79126e13ac7e72bdc980a64c7aa6c5b96d9d2fff8b5484a43aa0060c9ef6d575"
REGISTRY = "scripts/GPU/alphazero/e4_screen_reference.py"
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
    return {"game_id": game["game_id"], "family": game["family"], "seed": game["seed"]}


def game_starts(lines):
    """[(game_id, seed)] of every `game_start`, in order; nothing else of any line is read."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["event"] == "game_start":
            out.append((entry["game_id"], entry["seed"]))
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

def reconcile(plan, starts, games, recorded, drawn, registries, final):
    """Faults, and the block (or None). `games`: [seed_fields(file)] in plan order; `recorded`: the record's
    openings by family; `drawn`: the opening each plan opening seed draws, by family; `final`: (status,
    group_reaped)."""
    faults = []
    want = [(g["game_id"], g["seed"]) for g in plan["games"]]
    if len(starts) != len(set(starts)):
        faults.append("a game started more than once")
    if starts != want:
        faults.append(f"the {len(starts)} game starts are not the plan's {len(want)} games and seeds, in order")
    if games != [{k: g[k] for k in ("game_id", "family", "seed")} for g in plan["games"]]:
        faults.append("the game files' seed fields are not the plan's games")
    families = [str(o["family"]) for o in plan["openings"]]
    if sorted(recorded) != sorted(families):
        faults.append("the record's openings are not the plan's families")
    used_openings = [o["opening_seed"] for o in plan["openings"]
                     if str(o["family"]) in recorded and recorded[str(o["family"])] == drawn.get(str(o["family"]))]
    if len(used_openings) != len(families):
        faults.append("a recorded opening is not the one its plan seed draws")
    used = used_openings + [seed for gid, seed in starts if (gid, seed) in want]
    seeds = sorted(set(used))
    block = (seeds[0], seeds[-1] + 1) if seeds else None
    n = len(plan["openings"]) + len(plan["games"])
    if not seeds or seeds != list(range(*block)) or len(seeds) != n or len(used) != n:
        faults.append(f"the used seeds ({len(seeds)} unique of {len(used)}) are not one contiguous block of {n}")
    elif block != (plan["base"], plan["base"] + plan["n_slots"]):
        faults.append("the used seeds are not the plan's block")
    if block is not None:
        if registries["ACCOUNTED_SEED_INTERVALS"].count(block) != 1:
            faults.append("the block is not ACCOUNTED exactly as itself")
        for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
            if any(lo < block[1] and block[0] < hi for lo, hi in registries[name]):
                faults.append(f"seeds of the block are already in {name}")
    if final != ("EXITED", True):
        faults.append(f"the run is not finalized with its group reaped: {final}")
    return faults, block


# ─────────────────────────────── the controls ───────────────────────────────

def _world(families=3, lo=500):
    openings = [{"family": j, "opening_seed": lo + j} for j in range(families)]
    games = [{"game_id": f"f{j}-{c}", "family": j, "seed": lo + families + 2 * j + k}
             for j in range(families) for k, c in enumerate(("r", "b"))]
    plan = {"base": lo, "n_slots": 3 * families, "openings": openings, "games": games}
    starts = [(g["game_id"], g["seed"]) for g in games]
    files = [dict(g) for g in games]
    ops = {str(j): [[j, j]] for j in range(families)}
    regs = {"ACCOUNTED_SEED_INTERVALS": ((100, 132), (lo, lo + 3 * families)), "EXPOSED_SEED_INTERVALS": ((100, 132),),
            "RETIRED_SEED_INTERVALS": ((100, 132),), "TEST_ONLY_SEED_INTERVALS": ()}
    return plan, starts, files, ops, dict(ops), regs, ("EXITED", True)


def _controls():
    def ok(edit):
        w = list(_world())
        edit(w)
        return not reconcile(*w)[0]

    def setw(i, v):
        return lambda w: w.__setitem__(i, v)
    plan, starts, files, ops, _drawn, regs, _fin = _world()
    other_seed = [dict(f) for f in files]
    other_seed[2]["seed"] += 1
    gap_plan = dict(plan, n_slots=plan["n_slots"] + 1, games=[dict(g) for g in plan["games"]])
    gap_plan["games"][-1]["seed"] += 1                               # a gap INSIDE the plan's declared block
    gap_starts = [(g["game_id"], g["seed"]) for g in gap_plan["games"]]
    gap_files = [dict(g) for g in gap_plan["games"]]
    shifted = dict(plan, base=plan["base"] + 1)
    cases = [
        ("a sound world reconciles", lambda w: None, True),
        ("a game start missing", setw(1, starts[:-1]), False),
        ("a game started twice", setw(1, starts + starts[-1:]), False),
        ("the starts out of plan order", setw(1, [starts[1], starts[0]] + starts[2:]), False),
        ("a game started with another seed", setw(1, starts[:-1] + [(starts[-1][0], starts[-1][1] + 1)]), False),
        ("a game file missing", setw(2, files[:-1]), False),
        ("a game file's seed is not its plan entry's", setw(2, other_seed), False),
        ("a recorded opening is not the one its seed draws", setw(4, dict(ops, **{"1": [[9, 9]]})), False),
        ("a family's opening not recorded", setw(3, {k: v for k, v in ops.items() if k != "2"}), False),
        ("the used seeds are not one contiguous block",
         lambda w: (w.__setitem__(0, gap_plan), w.__setitem__(1, gap_starts), w.__setitem__(2, gap_files),
                    w.__setitem__(5, dict(regs, ACCOUNTED_SEED_INTERVALS=((100, 132), (500, 510))))), False),
        ("the used seeds are not the plan's declared block", setw(0, shifted), False),
        ("the block is not ACCOUNTED", setw(5, dict(regs, ACCOUNTED_SEED_INTERVALS=((100, 132),))), False),
        ("the block ACCOUNTED twice", setw(5, dict(regs, ACCOUNTED_SEED_INTERVALS=regs["ACCOUNTED_SEED_INTERVALS"] * 2)),
         False),
        ("a seed of the block already EXPOSED", setw(5, dict(regs, EXPOSED_SEED_INTERVALS=((505, 506),))), False),
        ("a seed of the block already RETIRED", setw(5, dict(regs, RETIRED_SEED_INTERVALS=((499, 501),))), False),
        ("an adjacent EXPOSED interval is no overlap", setw(5, dict(regs, EXPOSED_SEED_INTERVALS=((490, 500),))), True),
        ("the run's group not reaped", setw(6, ("EXITED", False)), False),
        ("the run timed out", setw(6, ("TIMEOUT", True)), False),
    ]
    out = [(label, ok(edit) is want, f"expected {want}") for label, edit, want in cases]
    g = seed_fields({"game_id": "f0-r", "family": 0, "seed": 1, "moves": [[0, 0]], "winner": "red"})
    out.append(("seed_fields returns ONLY game_id, family and seed", g == {"game_id": "f0-r", "family": 0, "seed": 1}, ""))
    lines = [json.dumps({"event": "game_start", "game_id": "f0-r", "seed": 1, "opening": []}),
             json.dumps({"event": "move", "game_id": "f0-r", "move": [0, 0]}), ""]
    out.append(("game_starts reads only game_start entries", game_starts(lines) == [("f0-r", 1)], ""))
    return out


# ─────────────────────────────── the run ───────────────────────────────

def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _inputs():
    """Every input, read as named above; refuses unless all are committed, unmodified, pinned and hashed."""
    if os.path.lexists(ROOT / OUT):
        raise Refused(f"{OUT} exists: the derivation runs once (create-only)")
    if _git("status", "--porcelain").stdout.strip():
        raise Refused("the working tree is not clean: the derivation reads the COMMITTED state")
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import external_learning as EL
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    d = ROOT / RUN
    plan_raw = (ROOT / PLAN).read_bytes()
    if _sha(plan_raw) != PLAN_SHA256:
        raise Refused("the plan is not stage 1's committed plan")
    plan = json.loads(plan_raw)
    manifest = {}
    for line in (d / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        manifest[name] = sha
    names = ["record.json", "finalized.json", "progress.jsonl"] + [f"games/{g['game_id']}.json" for g in plan["games"]]
    paths = [ROOT / PLAN, ROOT / REGISTRY, d / "MANIFEST.sha256"] + [d / n for n in names]
    try:
        F.check_durable([str(p) for p in paths], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise Refused(f"an input is not committed and unmodified: {e}") from None
    raw = {}
    for n in names:
        raw[n] = (d / n).read_bytes()
        if manifest.get(n) != _sha(raw[n]):
            raise Refused(f"{n} is not the file the manifest hashed")
    record = json.loads(raw["record.json"])
    if record["steps"]["plan"]["sha256"] != PLAN_SHA256:
        raise Refused("the run ran another plan")
    fin = json.loads(raw["finalized.json"])
    recorded = record["steps"]["openings"]
    drawn = {str(o["family"]): [list(m) for m in EL.generate_opening(o["opening_seed"])] for o in plan["openings"]}
    return {"head": _git("rev-parse", "HEAD").stdout.strip(), "plan": plan,
            "starts": game_starts(raw["progress.jsonl"].decode("utf-8").splitlines()),
            "games": [seed_fields(json.loads(raw[f"games/{g['game_id']}.json"])) for g in plan["games"]],
            "recorded": recorded, "drawn": drawn,
            "registries": registry_intervals((ROOT / REGISTRY).read_text(encoding="utf-8")),
            "final": (fin["status"], fin["group_reaped"]), "exit": (fin["exit_code"], fin["verdict"]),
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
    say("EXTERNAL-ENGINE LEARNING STAGE 1 -- CLOSING SEED ACCOUNTING OF THE STAGE-1 BLOCK (read-only, outcome-blind)")
    say("=" * 96)
    say(f"HEAD {x['head']}; working tree clean; {x['files']} inputs committed and unmodified "
        f"(plan sha256 {PLAN_SHA256}, registry, manifest, record, finalized, progress, {len(x['games'])} game files), "
        "each run file the one the manifest hashed")
    faults, block = reconcile(x["plan"], x["starts"], x["games"], x["recorded"], x["drawn"], x["registries"],
                              x["final"])
    say(f"run (finalized status, group reaped): {x['final']}; exit code, verdict: {x['exit']}")
    say(f"game starts in progress.jsonl: {len(x['starts'])}, distinct {len(set(x['starts']))}; the plan's games: "
        f"{len(x['plan']['games'])}; game files read: {len(x['games'])}")
    same = sum(x["recorded"].get(k) == v for k, v in x["drawn"].items())
    say(f"openings: {len(x['recorded'])} recorded; {same} of {len(x['drawn'])} equal the opening their plan seed draws")
    say(f"the block the used seeds make: {block}; ACCOUNTED holds it as itself: "
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
    say("VERDICT: " + (f"RECONCILES -- all {block[1] - block[0]} seeds of {block} used (6 openings drawn, 12 games "
                       "started): the stage-1 block is EXPOSED WHOLE and RETIRED WHOLE" if good else
                       "DOES NOT RECONCILE -- no registry edit"))
    _write(ROOT / OUT)
    return 0 if good else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
