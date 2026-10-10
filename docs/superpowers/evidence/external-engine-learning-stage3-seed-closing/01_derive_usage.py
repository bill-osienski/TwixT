"""EXTERNAL-ENGINE LEARNING STAGE 3 -- THE CLOSING SEED ACCOUNTING of the stage-3 game seeds, DERIVED from
committed evidence.

Card `docs/superpowers/2026-10-08-external-engine-learning-viability.md` (seeds: "actual exposure retirement");
the review rule for stage 3: seed use counts EVERY STARTED game, interrupted ones included, not only completed
ones; a held-out stop exposes zero game seeds. The run is committed at evidence/external-engine-learning-stage3/.
This script is READ-ONLY and OUTCOME-BLIND; of every input it reads only what is named here:
  * progress.jsonl: each line's `event` and, for a `game_start` only, its game and seed;
  * each replay file the manifest lists under games/, through `seed_fields`: its game_idx and seed. No other field;
  * record.json: steps.plan.sha256 only;
  * finalized.json: status, group_reaped, exit_code and verdict; MANIFEST.sha256: every line;
  * the committed plan (pinned by sha256) and the registry's four intervals, by parsing.
THE RECONCILIATION: the game starts are distinct and are exactly the plan's first k slots, games and seeds in
order (k = 0 after a held-out stop); every replay file belongs to a started game and carries that slot's seed;
the plan's block is ACCOUNTED exactly as itself and overlaps neither EXPOSED nor RETIRED; the run finalized with
its group reaped (EXITED, or TIMEOUT -- an interrupted cohort still used the seeds it started); every input
committed, unmodified and the file the manifest hashed.
RECONCILES (exit 0): EXPOSED = the k started seeds [base, base + k) (none when k = 0); RETIRED = the plan's
whole block (no retry, extension or replacement seed). Otherwise DOES NOT RECONCILE (1) -- no registry edit.
Refused before anything is written (4). One run; create-only record 02_derive_usage_run.txt.

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
HERE = "docs/superpowers/evidence/external-engine-learning-stage3-seed-closing"
OUT = f"{HERE}/02_derive_usage_run.txt"
RUN = "docs/superpowers/evidence/external-engine-learning-stage3"
PLAN = "docs/superpowers/evidence/external-engine-learning-plans/stage3.json"
PLAN_SHA256 = "0eab28b084514e34f4454e185e909072d5816f62ee2dd88ca3ba73c20cba5a41"
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

def seed_fields(replay):
    """The ONLY fields of a replay the derivation reads."""
    return {"game": replay["game_idx"], "seed": replay["seed"]}


def game_starts(lines):
    """[(game, seed)] of every `game_start`, in order; nothing else of any line is read."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["event"] == "game_start":
            out.append((entry["game"], entry["seed"]))
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

def reconcile(plan, starts, replays, registries, final):
    """Faults, the EXPOSED interval (None when no game started) and the RETIRED interval (the plan's block).
    `replays`: [seed_fields(file)]; `final`: (status, group_reaped)."""
    faults = []
    slots = [(g["game"], g["seed"]) for g in plan["games"]]
    block = (plan["base"], plan["base"] + plan["n_slots"])
    if len(starts) != len(set(starts)):
        faults.append("a game started more than once")
    if starts != slots[:len(starts)]:
        faults.append(f"the {len(starts)} game starts are not the plan's first {len(starts)} slots, in order")
    if [s for _g, s in slots] != list(range(*block)):
        faults.append("the plan's slot seeds are not its block, in order")
    started = set(starts)
    if any((r["game"], r["seed"]) not in started for r in replays) or len(replays) != len({r["game"] for r in replays}):
        faults.append("a replay file is not a started game with its slot's seed")
    k = len(starts)
    exposed = (block[0], block[0] + k) if k else None
    if registries["ACCOUNTED_SEED_INTERVALS"].count(block) != 1:
        faults.append("the plan's block is not ACCOUNTED exactly as itself")
    for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
        if any(lo < block[1] and block[0] < hi for lo, hi in registries[name]):
            faults.append(f"seeds of the block are already in {name}")
    if final[0] not in ("EXITED", "TIMEOUT") or final[1] is not True:
        faults.append(f"the run is not finalized with its group reaped: {final}")
    return faults, exposed, block


# ─────────────────────────────── the controls ───────────────────────────────

def _world(n=6, lo=500, k=None):
    k = n if k is None else k
    games = [{"game": g, "pair": g // 2, "seed": lo + g} for g in range(n)]
    plan = {"base": lo, "n_slots": n, "games": games}
    starts = [(g["game"], g["seed"]) for g in games[:k]]
    replays = [{"game": g, "seed": s} for g, s in starts]
    regs = {"ACCOUNTED_SEED_INTERVALS": ((100, 132), (lo, lo + n)), "EXPOSED_SEED_INTERVALS": ((100, 132),),
            "RETIRED_SEED_INTERVALS": ((100, 132),), "TEST_ONLY_SEED_INTERVALS": ()}
    return plan, starts, replays, regs, ("EXITED", True)


def _controls():
    def faults(w, edit=None):
        w = list(w)
        if edit:
            edit(w)
        return reconcile(*w)

    def setw(i, v):
        return lambda w: w.__setitem__(i, v)
    plan, starts, replays, regs, _fin = _world()
    out = []
    f, exp, ret = faults(_world())
    out.append(("a complete cohort reconciles: all seeds EXPOSED, the block RETIRED",
                not f and exp == (500, 506) and ret == (500, 506), str((f, exp, ret))))
    f, exp, ret = faults(_world(k=3), setw(4, ("TIMEOUT", True)))
    out.append(("an interrupted cohort (3 started, timed out) exposes the 3 started seeds, retires the block",
                not f and exp == (500, 503) and ret == (500, 506), str((f, exp, ret))))
    w = _world(k=3)
    f, exp, ret = faults(w, setw(2, w[2][:2]))
    out.append(("a started game with NO replay (interrupted mid-game) is still EXPOSED", not f and exp == (500, 503),
                str((f, exp))))
    f, exp, ret = faults(_world(k=0))
    out.append(("a held-out stop (no game started) exposes zero seeds, retires the block",
                not f and exp is None and ret == (500, 506), str((f, exp, ret))))
    cases = [
        ("a game started twice", setw(1, starts + starts[-1:])),
        ("the starts out of plan order", setw(1, [starts[1], starts[0]] + starts[2:])),
        ("a game skipped (the starts are not a prefix)", setw(1, starts[:2] + starts[3:])),
        ("a game started with another seed", setw(1, starts[:-1] + [(starts[-1][0], starts[-1][1] + 1)])),
        ("a replay of a game never started", setw(1, starts[:-1])),
        ("a replay with another seed", setw(2, replays[:-1] + [{"game": 5, "seed": 999}])),
        ("two replays of one game", setw(2, replays + replays[-1:])),
        ("the plan's slots not its block", setw(0, dict(plan, base=plan["base"] + 1))),
        ("the block not ACCOUNTED", setw(3, dict(regs, ACCOUNTED_SEED_INTERVALS=((100, 132),)))),
        ("the block ACCOUNTED twice", setw(3, dict(regs, ACCOUNTED_SEED_INTERVALS=regs["ACCOUNTED_SEED_INTERVALS"] * 2))),
        ("a seed of the block already EXPOSED", setw(3, dict(regs, EXPOSED_SEED_INTERVALS=((505, 506),)))),
        ("a seed of the block already RETIRED", setw(3, dict(regs, RETIRED_SEED_INTERVALS=((499, 501),)))),
        ("the run's group not reaped", setw(4, ("EXITED", False))),
        ("the run's status unknown", setw(4, ("UNKNOWN", True))),
    ]
    out += [(label, bool(faults(_world(), edit)[0]), "expected a fault") for label, edit in cases]
    f = faults(_world(), setw(3, dict(regs, EXPOSED_SEED_INTERVALS=((490, 500),))))[0]
    out.append(("an adjacent EXPOSED interval is no overlap", not f, str(f)))
    g = seed_fields({"game_idx": 0, "seed": 1, "task_id": 0, "moves": [[0, 0]], "winner": "red"})
    out.append(("seed_fields returns ONLY game and seed", g == {"game": 0, "seed": 1}, ""))
    lines = [json.dumps({"event": "game_start", "game": 0, "seed": 1, "red": "candidate"}),
             json.dumps({"event": "game_end", "game": 0, "winner_agent_id": "candidate"}), ""]
    out.append(("game_starts reads only game_start entries", game_starts(lines) == [(0, 1)], ""))
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
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    d = ROOT / RUN
    plan_raw = (ROOT / PLAN).read_bytes()
    if _sha(plan_raw) != PLAN_SHA256:
        raise Refused("the plan is not stage 3's committed plan")
    plan = json.loads(plan_raw)
    manifest = {}
    for line in (d / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        manifest[name] = sha
    names = ["record.json", "finalized.json", "progress.jsonl"] + sorted(n for n in manifest if n.startswith("games/"))
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
    if json.loads(raw["record.json"])["steps"]["plan"]["sha256"] != PLAN_SHA256:
        raise Refused("the run ran another plan")
    fin = json.loads(raw["finalized.json"])
    return {"head": _git("rev-parse", "HEAD").stdout.strip(), "plan": plan,
            "starts": game_starts(raw["progress.jsonl"].decode("utf-8").splitlines()),
            "replays": [seed_fields(json.loads(raw[n])) for n in names if n.startswith("games/")],
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
    say("EXTERNAL-ENGINE LEARNING STAGE 3 -- CLOSING SEED ACCOUNTING OF THE STAGE-3 GAME SEEDS (read-only, outcome-blind)")
    say("=" * 96)
    say(f"HEAD {x['head']}; working tree clean; {x['files']} inputs committed and unmodified "
        f"(plan sha256 {PLAN_SHA256}, registry, manifest, record, finalized, progress, {len(x['replays'])} replay "
        "files), each run file the one the manifest hashed")
    faults, exposed, retired = reconcile(x["plan"], x["starts"], x["replays"], x["registries"], x["final"])
    say(f"run (finalized status, group reaped): {x['final']}; exit code, verdict: {x['exit']}")
    say(f"game starts in progress.jsonl: {len(x['starts'])}, distinct {len(set(x['starts']))}; the plan's slots: "
        f"{len(x['plan']['games'])}; replay files read: {len(x['replays'])}")
    say(f"seeds used (every started game): {exposed}; the plan's block: {retired}")
    say(f"faults: {faults if faults else 'none'}")
    say("\nCONTROLS (synthetic)")
    say("-" * 96)
    results = _controls()
    for label, ok, detail in results:
        say(f"  {'PASS' if ok else 'FAIL':6s} {label}{'  [' + detail + ']' if detail and not ok else ''}")
    passed = sum(ok for _l, ok, _d in results)
    say(f"\ncontrols passed: {passed}/{len(results)}")
    good = not faults and passed == len(results) > 0
    say("VERDICT: " + (f"RECONCILES -- EXPOSED {exposed if exposed else 'nothing (no game started)'}; "
                       f"RETIRED the whole block {retired}" if good else "DOES NOT RECONCILE -- no registry edit"))
    _write(ROOT / OUT)
    return 0 if good else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
