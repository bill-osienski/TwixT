"""ROOT-HALVING MATCH -- THE SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

The closing step of evidence/root-halving-match/01_match.py. A copy of
evidence/mirror-averaging-match-seed-closing/01_derive_usage.py; the diff: the paths and the titles. Fixed BEFORE the run:
  * RETIRE the whole block [BASE, BASE + 32), whatever happened.
  * EXPOSE only the started-game prefix: slots 0..k-1, where k = the number of `game_start` events in
    progress.jsonl (logged BEFORE each game is played, so an interrupted game counts as started). A run
    that stopped before either game (REFUSED, a loading error, a timeout during loading) exposes NEITHER.
Reads only: the plan, progress.jsonl's event names and slots, outer.json (outer exit status and
survivors), record.json's verdict if it exists, and the registry. Never a score, winner or move.

RECONCILES only if: every input is committed and unmodified; outer.json shows termination confirmed
(no survivors); progress starts with `start`; the game_start slots are exactly 0..k-1 in order, k <= 32;
outer exit 0 has k = 32 and verdict COMPLETED; a REFUSED (4) run has k = 0; the block is ACCOUNTED exactly
once and in neither EXPOSED nor RETIRED. Then the ONE closing edit appends [BASE, BASE + k) to EXPOSED
(only if k > 0) and [BASE, BASE + 32) to RETIRED, each last.

One create-only record: 02_derive_usage_run.txt. Exit 0 RECONCILES / 1 does not / 4 refused.
This script holds no seed literal.

    python -B <this file>
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = "docs/superpowers/evidence/root-halving-match-seed-closing"
RUN = "docs/superpowers/evidence/root-halving-match"
INPUTS = [f"{RUN}/plan.json", f"{RUN}/progress.jsonl", f"{RUN}/outer.json"]
OUT = f"{HERE}/02_derive_usage_run.txt"
N_SLOTS = 32


def reconcile(plan, events, outer, verdict, regs):
    """(faults, expose, retire) -- expose/retire are (lo, hi) or None. Pure; the record's controls drive it."""
    faults = []
    base = plan.get("base")
    if plan.get("n_slots") != N_SLOTS or type(base) is not int:
        return [f"the plan is not {N_SLOTS} slots from an int base"], None, None
    block = (base, base + N_SLOTS)
    if outer.get("survivors", "").strip():
        faults.append(f"termination NOT confirmed: survivors {outer['survivors']!r}")
    if not events or events[0].get("event") != "start":
        faults.append("progress does not begin with start")
    slots = [e.get("slot") for e in events if e.get("event") == "game_start"]
    k = len(slots)
    if slots != list(range(k)) or k > N_SLOTS:
        faults.append(f"game_start slots {slots} are not a prefix 0..k-1 of the {N_SLOTS} slots")
    status = outer.get("outer_exit")
    if status == 0 and (k != N_SLOTS or verdict != "COMPLETED"):
        faults.append(f"outer exit 0 but {k} games started, verdict {verdict!r}")
    if status == 4 and k != 0:
        faults.append(f"a REFUSED run started {k} games")
    acc, exp, ret = (regs[n] for n in ("ACCOUNTED", "EXPOSED", "RETIRED"))
    if acc.count(block) != 1:
        faults.append("the block is not ACCOUNTED exactly once")
    if any(lo < block[1] and block[0] < hi for lo, hi in exp + ret):
        faults.append("a seed of the block is already EXPOSED or RETIRED")
    if faults:
        return faults, None, None
    return [], ((base, base + k) if k else None), block


def _controls(base):
    """Synthetic inputs: the rule must give the fixed answer and refuse every wrong shape."""
    n = N_SLOTS
    plan = {"base": base, "n_slots": n}
    regs = {"ACCOUNTED": [(base, base + n)], "EXPOSED": [], "RETIRED": []}
    st = [{"event": "start"}]
    g = lambda s: {"event": "game_start", "slot": s}
    upto = lambda k: st + [g(s) for s in range(k)]            # the started prefix 0..k-1
    ok = {"outer_exit": 0, "survivors": ""}
    cases = [(f"completed: {n} starts, expose all, retire all", upto(n), ok, "COMPLETED", regs, (base, base + n)),
             ("REFUSED before any game: expose NONE, retire all", st, {**ok, "outer_exit": 4}, "REFUSED", regs,
              None),
             ("timeout during loading: expose none", st, {**ok, "outer_exit": 124}, None, regs, None),
             ("timeout in game 0: expose slot 0 only", upto(1), {**ok, "outer_exit": 124}, None, regs,
              (base, base + 1)),
             ("timeout in game 20: expose slots 0-20", upto(21), {**ok, "outer_exit": 124}, None, regs,
              (base, base + 21)),
             (f"kill in the last game: expose all {n}", upto(n), {**ok, "outer_exit": 137}, None, regs,
              (base, base + n)),
             (f"integrity stop in game {n - 1}: expose all", upto(n), {**ok, "outer_exit": 2}, "STOP_INTEGRITY",
              regs, (base, base + n)),
             ("survivors: refused", upto(5), {**ok, "outer_exit": 137, "survivors": "123 "}, None, regs, "FAULT"),
             ("slots out of order: refused", st + [g(0), g(2)], {**ok, "outer_exit": 124}, None, regs, "FAULT"),
             ("a slot started twice: refused", upto(3) + [g(2)], {**ok, "outer_exit": 3}, "STOP_ERROR", regs, "FAULT"),
             (f"more than {n} starts: refused", upto(n + 1), {**ok, "outer_exit": 3}, "STOP_ERROR", regs, "FAULT"),
             (f"exit 0 with {n - 1} starts: refused", upto(n - 1), ok, "COMPLETED", regs, "FAULT"),
             ("REFUSED after a game: refused", upto(1), {**ok, "outer_exit": 4}, "REFUSED", regs, "FAULT"),
             ("block not ACCOUNTED: refused", st, {**ok, "outer_exit": 4}, "REFUSED", {**regs, "ACCOUNTED": []},
              "FAULT"),
             ("block already RETIRED in part: refused", st, {**ok, "outer_exit": 4}, "REFUSED",
              {**regs, "RETIRED": [(base + n - 1, base + n)]}, "FAULT")]
    out = []
    for label, events, outer, verdict, r, want in cases:
        faults, expose, retire = reconcile(plan, events, outer, verdict, r)
        good = bool(faults) if want == "FAULT" else (not faults and expose == want and retire == (base, base + n))
        out.append((label, good, "; ".join(faults) or f"expose {expose}, retire {retire}"))
    return out


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def main():
    out = ROOT / OUT
    if out.exists():
        print(f"REFUSED: {OUT} exists (create-only)", file=sys.stderr)
        return 4
    inputs = INPUTS + ([f"{RUN}/record.json"] if (ROOT / RUN / "record.json").exists() else [])
    for rel in inputs:
        if _git("ls-files", "--error-unmatch", "--", rel).returncode or _git("diff", "--quiet", "HEAD", "--", rel).returncode:
            print(f"REFUSED: {rel} is absent, uncommitted or modified", file=sys.stderr)
            return 4
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import e4_screen_reference as REG
    plan = json.loads((ROOT / INPUTS[0]).read_text())
    events = [json.loads(line) for line in (ROOT / INPUTS[1]).read_text().splitlines() if line.strip()]
    outer = json.loads((ROOT / INPUTS[2]).read_text())
    verdict = json.loads((ROOT / RUN / "record.json").read_text()).get("verdict") if len(inputs) > 3 else None
    regs = {n: [tuple(iv) for iv in getattr(REG, f"{n}_SEED_INTERVALS")] for n in ("ACCOUNTED", "EXPOSED", "RETIRED")}
    faults, expose, retire = reconcile(plan, events, outer, verdict, regs)
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    lines = ["ROOT-HALVING MATCH -- SEED-USAGE DERIVATION (closing)", "=" * 96,
             f"HEAD {_git('rev-parse', 'HEAD').stdout.strip()}; working tree {'DIRTY' if dirty else 'clean'}",
             *(f"input {rel} (committed, unmodified)" for rel in inputs),
             f"outer exit {outer.get('outer_exit')}; survivors {outer.get('survivors')!r}; verdict {verdict!r}",
             f"game starts (slots): {[e.get('slot') for e in events if e.get('event') == 'game_start']}"]
    controls = _controls(plan.get("base", 0) if type(plan.get("base")) is int else 0)
    lines += ["", "CONTROLS"] + [f"  {'PASS' if ok else 'FAIL':6s} {label}  [{d}]" for label, ok, d in controls]
    passed = sum(ok for _l, ok, _d in controls)
    lines.append(f"\ncontrols passed: {passed}/{len(controls)}")
    good = not faults and passed == len(controls) and not dirty
    faults = faults + (["the working tree is dirty"] if dirty else [])
    lines.append("VERDICT: " + (f"RECONCILES -- EXPOSE {list(expose) if expose else 'NOTHING'}; RETIRE {list(retire)}"
                                if good else "DOES NOT RECONCILE -- STOP: " + "; ".join(faults or ["a control failed"])))
    print("\n".join(lines))
    with open(out, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(lines) + "\n")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
