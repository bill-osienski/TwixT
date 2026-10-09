"""EXTERNAL-ENGINE LEARNING STAGE 2 -- THE CLOSING SEED ACCOUNTING of the one training seed, DERIVED from
committed evidence.

The registry comment of the seed: exposure follows actual use, in a later closing edit. The stage-2 run is
committed at evidence/external-engine-learning-stage2/. The seed's ONLY use is the training sampler. This
script is READ-ONLY; of every input it reads only what is named here:
  * record.json: verdict, steps.plan.sha256, steps.schedule_sha256, steps.inputs.train_rows and
    steps.training.steps;
  * progress.jsonl: each line's `event` and, for a `train_step` only, its step number;
  * finalized.json: status and group_reaped; MANIFEST.sha256: every line;
  * the committed plan (pinned by sha256) and the registry's four intervals, by parsing.
THE RECONCILIATION: the run ran THIS plan; the schedule the plan's seed draws (recomputed here with the run's
own `sampling_schedule`) hashes to the record's schedule hash, so the seed was drawn; the logged train steps
are exactly 0 .. STEPS-1, once each, as many as the record's; the seed's one-seed block is ACCOUNTED exactly
as itself and in neither EXPOSED nor RETIRED; the run finalized with its group reaped; every input
committed, unmodified and the file the manifest hashed.
RECONCILES (exit 0): the seed was used, so its block is EXPOSED WHOLE and RETIRED WHOLE. Otherwise DOES NOT
RECONCILE (1) -- no registry edit. Refused before anything is written (4). One run; create-only record
02_derive_usage_run.txt.

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
HERE = "docs/superpowers/evidence/external-engine-learning-stage2-seed-closing"
OUT = f"{HERE}/02_derive_usage_run.txt"
RUN = "docs/superpowers/evidence/external-engine-learning-stage2"
PLAN = "docs/superpowers/evidence/external-engine-learning-plans/stage2.json"
PLAN_SHA256 = "1420bda3ec6e6ced66d6f7e2cc052fcb4ad2d4234b1b0ba46b9abae4e1167ebe"
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


def _jsha(obj):
    return _sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


# ─────────────────────────────── the whitelisted reads ───────────────────────────────

def record_fields(record):
    """The ONLY fields of record.json the derivation reads."""
    steps = record["steps"]
    return {"verdict": record["verdict"], "plan_sha256": steps["plan"]["sha256"],
            "schedule_sha256": steps["schedule_sha256"], "train_rows": steps["inputs"]["train_rows"],
            "steps": steps["training"]["steps"]}


def train_steps(lines):
    """[step] of every `train_step`, in order; nothing else of any line is read."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["event"] == "train_step":
            out.append(entry["step"])
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

def reconcile(plan, rec, drawn_sha, logged, n_steps, registries, final):
    """Faults, and the block (or None). `rec`: record_fields; `drawn_sha`: the hash of the schedule the plan's
    seed draws; `logged`: the train_step numbers; `n_steps`: the fixed budget; `final`: (status, reaped)."""
    faults = []
    if rec["plan_sha256"] != PLAN_SHA256:
        faults.append("the run ran another plan")
    if drawn_sha != rec["schedule_sha256"]:
        faults.append("the schedule the plan's seed draws is not the run's")
    if logged != list(range(n_steps)) or rec["steps"] != n_steps:
        faults.append(f"the logged train steps ({len(logged)}) are not 0 .. {n_steps - 1} once each, as recorded")
    seed = plan["seed"]
    block = (seed, seed + 1) if type(seed) is int and plan.get("n_slots") == 1 else None
    if block is None:
        faults.append("the plan does not name exactly one int seed")
    else:
        if registries["ACCOUNTED_SEED_INTERVALS"].count(block) != 1:
            faults.append("the block is not ACCOUNTED exactly as itself")
        for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
            if any(lo < block[1] and block[0] < hi for lo, hi in registries[name]):
                faults.append(f"the seed is already in {name}")
    if final != ("EXITED", True):
        faults.append(f"the run is not finalized with its group reaped: {final}")
    return faults, block


# ─────────────────────────────── the controls ───────────────────────────────

def _world(seed=500, n=4):
    plan = {"seed": seed, "n_slots": 1}
    rec = {"verdict": "PASS", "plan_sha256": PLAN_SHA256, "schedule_sha256": "s" * 64, "train_rows": 9, "steps": n}
    regs = {"ACCOUNTED_SEED_INTERVALS": ((100, 132), (seed, seed + 1)), "EXPOSED_SEED_INTERVALS": ((100, 132),),
            "RETIRED_SEED_INTERVALS": ((100, 132),), "TEST_ONLY_SEED_INTERVALS": ()}
    return plan, rec, "s" * 64, list(range(n)), n, regs, ("EXITED", True)


def _controls():
    def ok(edit):
        w = list(_world())
        edit(w)
        return not reconcile(*w)[0]

    def setw(i, v):
        return lambda w: w.__setitem__(i, v)
    plan, rec, _s, logged, _n, regs, _f = _world()
    cases = [
        ("a sound world reconciles", lambda w: None, True),
        ("the run ran another plan", setw(1, {**rec, "plan_sha256": "0" * 64}), False),
        ("the seed draws another schedule", setw(2, "t" * 64), False),
        ("a train step missing", setw(3, logged[:-1]), False),
        ("a train step logged twice", setw(3, logged + logged[-1:]), False),
        ("the steps out of order", setw(3, [1, 0] + logged[2:]), False),
        ("the record counts other steps", setw(1, {**rec, "steps": 3}), False),
        ("the plan names no int seed", setw(0, {**plan, "seed": 500.0}), False),
        ("the plan names two slots", setw(0, {**plan, "n_slots": 2}), False),
        ("the seed is not ACCOUNTED", setw(5, dict(regs, ACCOUNTED_SEED_INTERVALS=((100, 132),))), False),
        ("the seed ACCOUNTED inside a wider interval", setw(5, dict(regs, ACCOUNTED_SEED_INTERVALS=((499, 502),))),
         False),
        ("the seed already EXPOSED", setw(5, dict(regs, EXPOSED_SEED_INTERVALS=((500, 501),))), False),
        ("the seed already RETIRED", setw(5, dict(regs, RETIRED_SEED_INTERVALS=((490, 510),))), False),
        ("an adjacent EXPOSED interval is no overlap", setw(5, dict(regs, EXPOSED_SEED_INTERVALS=((501, 510),))), True),
        ("the run's group not reaped", setw(6, ("EXITED", None)), False),
        ("the run timed out", setw(6, ("TIMEOUT", True)), False),
    ]
    out = [(label, ok(edit) is want, f"expected {want}") for label, edit, want in cases]
    lines = [json.dumps({"event": "train_step", "step": 0, "loss": 1.0}),
             json.dumps({"event": "step", "step": "training", "value": {"steps": 9}}), ""]
    out.append(("train_steps reads only train_step entries", train_steps(lines) == [0], ""))
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
    from scripts.GPU.alphazero import external_learning_stage2 as S2
    from scripts.GPU.alphazero import h4_pilot_feasibility as F
    d = ROOT / RUN
    names = ["record.json", "finalized.json", "progress.jsonl"]
    paths = [ROOT / PLAN, ROOT / REGISTRY, d / "MANIFEST.sha256"] + [d / n for n in names]
    try:
        F.check_durable([str(p) for p in paths], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise Refused(f"an input is not committed and unmodified: {e}") from None
    plan_raw = (ROOT / PLAN).read_bytes()
    if _sha(plan_raw) != PLAN_SHA256:
        raise Refused("the plan is not stage 2's committed plan")
    plan = json.loads(plan_raw)
    manifest = {name: sha for sha, name in (line.split("  ", 1) for line in
                                            (d / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines())}
    raw = {}
    for n in names:
        raw[n] = (d / n).read_bytes()
        if manifest.get(n) != _sha(raw[n]):
            raise Refused(f"{n} is not the file the manifest hashed")
    fin = json.loads(raw["finalized.json"])
    rec = record_fields(json.loads(raw["record.json"]))
    return {"head": _git("rev-parse", "HEAD").stdout.strip(), "plan": plan,
            "rec": rec, "drawn_sha": _jsha(S2.sampling_schedule(plan["seed"], rec["train_rows"]).tolist()),
            "logged": train_steps(raw["progress.jsonl"].decode("utf-8").splitlines()), "n_steps": S2.STEPS,
            "registries": registry_intervals((ROOT / REGISTRY).read_text(encoding="utf-8")),
            "final": (fin["status"], fin["group_reaped"]), "files": len(paths)}


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
    say("EXTERNAL-ENGINE LEARNING STAGE 2 -- CLOSING SEED ACCOUNTING OF THE TRAINING SEED (read-only)")
    say("=" * 96)
    say(f"HEAD {x['head']}; working tree clean; {x['files']} inputs committed and unmodified (plan sha256 "
        f"{PLAN_SHA256}, registry, manifest, record, finalized, progress), each run file the one the manifest hashed")
    faults, block = reconcile(x["plan"], x["rec"], x["drawn_sha"], x["logged"], x["n_steps"], x["registries"],
                              x["final"])
    say(f"run (finalized status, group reaped): {x['final']}; record verdict: {x['rec']['verdict']}")
    say(f"the schedule the plan's seed draws: sha256 {x['drawn_sha']}; the record's: {x['rec']['schedule_sha256']}")
    say(f"train steps logged: {len(x['logged'])} (0 .. {max(x['logged']) if x['logged'] else '-'}); recorded: "
        f"{x['rec']['steps']}; the fixed budget: {x['n_steps']}")
    say(f"the block of the seed: {block}; ACCOUNTED holds it as itself: "
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
    say("VERDICT: " + (f"RECONCILES -- the seed of {block} drew the run's whole schedule ({x['n_steps']} steps): "
                       "its block is EXPOSED WHOLE and RETIRED WHOLE" if good else
                       "DOES NOT RECONCILE -- no registry edit"))
    _write(ROOT / OUT)
    return 0 if good else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
