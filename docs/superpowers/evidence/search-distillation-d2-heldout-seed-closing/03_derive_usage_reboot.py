"""SEARCH DISTILLATION D2 HELD-OUT SET -- THE SEED-USAGE DERIVATION, AMENDED 2026-10-25 FOR A REBOOT INTERRUPTION.

(read-only, outcome-blind). Approved by the user on 2026-10-25 after the one authorized generation run was cut off by
a macOS kernel panic and reboot. 01_derive_usage.py (fixed before the run, b887dd04) stays UNCHANGED and is NOT
run: it needs the wrapper's outer.json, which does not exist because the wrapper's EXIT trap never ran.

The reboot case, stated here and never synthesized:
  * the runtime exit status is UNKNOWN -- there is no outer.json, record.json or exit to read, and none is made up;
  * process absence was OBSERVED AFTER THE REBOOT (02_interruption_note.txt), NOT verified by the wrapper.

The rule is unchanged from 01:
  * RETIRE the whole block [BASE, BASE + 12), whatever happened.
  * EXPOSE only the started-game prefix: slots 0..k-1, where k = the number of `game_start` events in
    progress.jsonl (logged BEFORE each game is played, so an interrupted game counts as started).

Bound inputs, each committed, unmodified and equal to its pinned sha256: the plan, progress.jsonl and the
interruption note. From progress.jsonl it keeps only each event's name and `game` slot -- never a score, winner,
ply count or label. It reads no game file.

RECONCILES only if: every bound input matches; outer.json, record.json and heldout.json (the run's finalization
records and manifest) and the phase's finalized.json are all ABSENT; progress starts with `start`; the game_start
slots are exactly 0..k-1 in order, k <= 12 (12 starts is a reboot in the last game); the block is
ACCOUNTED exactly once and in neither EXPOSED nor RETIRED. Then the ONE closing edit appends [BASE, BASE + k) to
EXPOSED (only if k > 0) and [BASE, BASE + 12) to RETIRED, each last.

One create-only record: 04_derive_usage_reboot_run.txt. Exit 0 RECONCILES / 1 does not / 4 refused.
This script holds no seed literal.

    python -B <this file>
"""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = "docs/superpowers/evidence/search-distillation-d2-heldout-seed-closing"
RUN = "docs/superpowers/evidence/search-distillation-d2-heldout"
RUN_DIR = "docs/superpowers/evidence/search-distillation-d2-heldout-run"
BOUND = {f"{RUN_DIR}/plan.json": "95fc3e24367b252868181e54247322b6ec45822ea6c42ae2cae2dd4e89c5ed2f",
         f"{RUN}/progress.jsonl": "744daff38a47481567920f0dd20c257b35f3043400d16b63270ab85597fee8b5",
         f"{RUN_DIR}/02_interruption_note.txt": "0143aafd231cca9d617202d16a3b67b92d42d547329db1f9e6663a46b1573553"}
PHASE = pathlib.Path.home() / "twixt-cache/search-distillation-d2/heldout-tf32off-20261025T133651Z/phase_00"
MUST_BE_ABSENT = [ROOT / RUN_DIR / "outer.json", ROOT / RUN / "record.json", ROOT / RUN / "heldout.json",
                  PHASE / "finalized.json"]
OUT = f"{HERE}/04_derive_usage_reboot_run.txt"
N_SLOTS = 12


def blind(events):
    """Event names and slots only: everything else (winner, plies, seconds, seed) is dropped unread."""
    return [{k: e[k] for k in ("event", "game") if k in e} for e in events]


def reconcile(plan, events, digests_ok, present, regs):
    """(faults, expose, retire) -- expose/retire are (lo, hi) or None. Pure; the record's controls drive it."""
    faults = []
    base = plan.get("base")
    if plan.get("n_slots") != N_SLOTS or type(base) is not int:
        return [f"the plan is not {N_SLOTS} slots from an int base"], None, None
    block = (base, base + N_SLOTS)
    faults += [f"bound input does not match its pinned sha256: {rel}" for rel, ok in digests_ok.items() if not ok]
    faults += [f"not a reboot interruption: {p} exists" for p in present]
    if not events or events[0].get("event") != "start":
        faults.append("progress does not begin with start")
    slots = [e.get("game") for e in events if e.get("event") == "game_start"]
    k = len(slots)
    if slots != list(range(k)) or k > N_SLOTS:
        faults.append(f"game_start slots {slots} are not a prefix 0..k-1 of the {N_SLOTS} slots")
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
    ok = {rel: True for rel in BOUND}
    st = [{"event": "start"}]
    g = lambda s: {"event": "game_start", "game": s}
    e = lambda s: {"event": "game_end", "game": s}
    upto = lambda k: st + [x for s in range(k) for x in (g(s), e(s))][:-1]   # k starts, the last unfinished
    cases = [("reboot in game 1 (slot 0 finished): expose slots 0-1", upto(2), ok, [], regs, (base, base + 2)),
             ("reboot during loading: expose NONE, retire all", st, ok, [], regs, None),
             ("reboot in game 0: expose slot 0 only", upto(1), ok, [], regs, (base, base + 1)),
             (f"reboot in game {n - 1} (no manifest): expose all {n}", upto(n), ok, [], regs, (base, base + n)),
             (f"more than {n} starts: refused", upto(n + 1), ok, [], regs, "FAULT"),
             ("outer.json present: refused", upto(2), ok, ["outer.json"], regs, "FAULT"),
             ("record.json present: refused", upto(2), ok, ["record.json"], regs, "FAULT"),
             ("heldout.json (manifest) present: refused", upto(2), ok, ["heldout.json"], regs, "FAULT"),
             ("finalized.json present: refused", upto(2), ok, ["finalized.json"], regs, "FAULT"),
             ("progress hash mismatch: refused", upto(2), {**ok, f"{RUN}/progress.jsonl": False}, [], regs, "FAULT"),
             ("note hash mismatch: refused", upto(2), {**ok, f"{RUN_DIR}/02_interruption_note.txt": False}, [],
              regs, "FAULT"),
             ("no start event: refused", upto(2)[1:], ok, [], regs, "FAULT"),
             ("slots out of order: refused", st + [g(1), g(0)], ok, [], regs, "FAULT"),
             ("a slot started twice: refused", upto(2) + [g(1)], ok, [], regs, "FAULT"),
             ("block not ACCOUNTED: refused", upto(2), ok, [], {**regs, "ACCOUNTED": []}, "FAULT"),
             ("block already EXPOSED in part: refused", upto(2), ok, [], {**regs, "EXPOSED": [(base, base + 1)]},
              "FAULT")]
    out = []
    for label, events, digests_ok, present, r, want in cases:
        faults, expose, retire = reconcile(plan, blind(events), digests_ok, present, r)
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
    for rel in BOUND:
        if _git("ls-files", "--error-unmatch", "--", rel).returncode or _git("diff", "--quiet", "HEAD", "--", rel).returncode:
            print(f"REFUSED: {rel} is absent, uncommitted or modified", file=sys.stderr)
            return 4
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import e4_screen_reference as REG
    raw = {rel: (ROOT / rel).read_bytes() for rel in BOUND}
    digests_ok = {rel: hashlib.sha256(b).hexdigest() == BOUND[rel] for rel, b in raw.items()}
    present = [str(p) for p in MUST_BE_ABSENT if p.exists()]
    plan = json.loads(raw[f"{RUN_DIR}/plan.json"])
    events = blind([json.loads(line) for line in raw[f"{RUN}/progress.jsonl"].decode().splitlines() if line.strip()])
    regs = {n: [tuple(iv) for iv in getattr(REG, f"{n}_SEED_INTERVALS")] for n in ("ACCOUNTED", "EXPOSED", "RETIRED")}
    faults, expose, retire = reconcile(plan, events, digests_ok, present, regs)
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    lines = ["SEARCH DISTILLATION D2 HELD-OUT SET -- SEED-USAGE DERIVATION, AMENDED FOR A REBOOT (closing)", "=" * 96,
             f"HEAD {_git('rev-parse', 'HEAD').stdout.strip()}; working tree {'DIRTY' if dirty else 'clean'}",
             *(f"input {rel} (committed, unmodified; sha256 {'MATCHES' if digests_ok[rel] else 'DIFFERS'})"
               for rel in BOUND),
             *(f"absent {p}: {'NO -- PRESENT' if str(p) in present else 'yes'}" for p in MUST_BE_ABSENT),
             "termination: interrupted by a kernel panic and reboot; runtime exit status UNKNOWN; process absence "
             "observed after the reboot (02_interruption_note.txt), NOT wrapper-verified",
             f"events read (names and slots only): {[e['event'] for e in events]}",
             f"game starts (slots): {[e.get('game') for e in events if e.get('event') == 'game_start']}"]
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
