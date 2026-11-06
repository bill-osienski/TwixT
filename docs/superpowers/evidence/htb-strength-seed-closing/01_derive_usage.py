"""THE H-TB CONFIRMATION TEST -- THE SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

Plan docs/superpowers/2026-11-06-htb-strength-plan.md §8 step 5 (`htb_strength run strength`, 220 games). BUILT ONLY (ledger, 2026-11-06): running it
is a separate approval. A copy of evidence/htb-screen-seed-closing/01_derive_usage.py (the H-TB screen's, run once,
whose description follows from its first line); the diff: the block size (220), the paths and the title. The rule
is unchanged: the test plays its games in order in one process and logs `game_start` before each, so only the
started prefix is exposed and the whole block is retired. Read the copied description's 64 as 220.

THE H-TB SCREEN -- THE SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

Plan docs/superpowers/2026-11-06-htb-screen-plan.md §7 step 5 (`htb_screen run screen`, 64 games). BUILT ONLY (ledger, 2026-11-06): running it is a
separate approval. A copy of evidence/combination-policy-cycle2-screen-seed-closing/01_derive_usage.py (the cycle-2
screen's, run once, whose description follows from its first line); the diff: the paths and the title. The block
size and the rule are unchanged: the screen plays its games in order in one process and logs `game_start` before each.

THE SECOND POLICY-IMPROVEMENT CYCLE -- THE SCREEN'S SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

Amendment docs/superpowers/2026-11-02-combination-policy-second-cycle-amendment.md §6 step 1 (`run cycle2_screen`, 64 games). BUILT ONLY (ledger, 2026-11-02): running it is a separate approval. A copy of
evidence/combination-policy-screen-seed-closing/01_derive_usage.py (the first cycle's, run once, whose description follows); the diff: the
paths and the title. The block size and the rule are unchanged.

THE COMBINATION POLICY PILOT -- THE SCREEN'S SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

Card docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 step 5 (`combination_policy_pilot run screen`,
64 games). BUILT ONLY (ledger, 2026-10-31, the user's review): running it is a separate approval. A copy of
evidence/mirror-halving-strength2-seed-closing/01_derive_usage.py (the strength match's attempt 2, whose description
follows); the diff: the block size (64), the paths and the title. The screen plays its games IN ORDER IN ONE PROCESS
through that match's loop and logs what the match logs -- `game_start` (game = slot) BEFORE each game -- so the
match's rule applies unchanged: the started games must be the prefix 0..k-1 IN ORDER, and only they are exposed. The
card's parallel-worker diff (generation's closing: starts as a set, a `claims` event) is for runs with workers, which
this is not. The `bg_check`, `loaded` and `game_end` events are read by name only. Read the copied description's 1018
as 64.

MIRROR AVERAGING + ROOT SEQUENTIAL HALVING -- THE STRENGTH ATTEMPT-2 SEED-USAGE DERIVATION for closing (read-only,
outcome-blind).

Amendment docs/superpowers/2026-10-27-mirror-halving-strength-attempt2-amendment.md (`run strength2`). A copy of
evidence/mirror-halving-strength-seed-closing/01_derive_usage.py (attempt 1's, run once at 376b75a3, whose description
follows); the diff: the paths and the titles. The `bg_check` events attempt 2 logs are read by name only, like every
other event that is not `start` or `game_start`.

Card docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md §5: the closing step of
scripts/GPU/alphazero/mirror_halving_match.py `run strength` (1018 games). A copy of
evidence/search-distillation-d2-heldout-attempt2-seed-closing/01_derive_usage.py; the diff: the block size (1018), the
paths and the titles. mirror_halving_match logs `game_start` (game = slot) BEFORE each game, as the rule below needs.
BUILT ONLY: running it is a separate approval. The copied description follows (read its 12 as 1018); attempt 1's
description follows within it.

The closing step of scripts/GPU/alphazero/search_distillation_d2_heldout.py (card
docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md §6). A copy of
evidence/search-distillation-d2-generate-seed-closing/01_derive_usage.py (Stage 2); the diff: the block size (12),
the paths, the mid-run control re-sized to 12 slots, and the titles. Fixed BEFORE the run:
  * RETIRE the whole block [BASE, BASE + 12), whatever happened.
  * EXPOSE only the started-game prefix: slots 0..k-1, where k = the number of `game_start` events in
    progress.jsonl (logged BEFORE each game is played, so an interrupted game counts as started). A run
    that stopped before either game (REFUSED, a loading error, a timeout during loading) exposes NEITHER.
Reads only: the plan, progress.jsonl's event names and slots, outer.json (outer exit status and
survivors), record.json's status if it exists, and the registry. Never a score, winner or move.

RECONCILES only if: every input is committed and unmodified; outer.json shows termination confirmed
(no survivors); progress starts with `start`; the game_start slots are exactly 0..k-1 in order, k <= 12;
outer exit 0 has k = 12 and status COMPLETED; a REFUSED (4) run has k = 0; the block is ACCOUNTED exactly
once and in neither EXPOSED nor RETIRED. Then the ONE closing edit appends [BASE, BASE + k) to EXPOSED
(only if k > 0) and [BASE, BASE + 12) to RETIRED, each last.

One create-only record: 02_derive_usage_run.txt. Exit 0 RECONCILES / 1 does not / 4 refused.
This script holds no seed literal.

    python -B <this file>
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = "docs/superpowers/evidence/htb-strength-seed-closing"
RUN = "docs/superpowers/evidence/htb-strength"
RUN_DIR = "docs/superpowers/evidence/htb-strength-run"
INPUTS = [f"{RUN_DIR}/plan.json", f"{RUN}/progress.jsonl", f"{RUN_DIR}/outer.json"]
OUT = f"{HERE}/02_derive_usage_run.txt"
N_SLOTS = 220


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
    slots = [e.get("game") for e in events if e.get("event") == "game_start"]
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
    g = lambda s: {"event": "game_start", "game": s}
    upto = lambda k: st + [g(s) for s in range(k)]            # the started prefix 0..k-1
    ok = {"outer_exit": 0, "survivors": ""}
    cases = [(f"completed: {n} starts, expose all, retire all", upto(n), ok, "COMPLETED", regs, (base, base + n)),
             ("REFUSED before any game: expose NONE, retire all", st, {**ok, "outer_exit": 4}, "REFUSED", regs,
              None),
             ("timeout during loading: expose none", st, {**ok, "outer_exit": 124}, None, regs, None),
             ("timeout in game 0: expose slot 0 only", upto(1), {**ok, "outer_exit": 124}, None, regs,
              (base, base + 1)),
             ("timeout in game 5: expose slots 0-5", upto(6), {**ok, "outer_exit": 124}, None, regs,
              (base, base + 6)),
             (f"kill in the last game: expose all {n}", upto(n), {**ok, "outer_exit": 137}, None, regs,
              (base, base + n)),
             (f"VOID in game {n - 1}: expose all", upto(n), {**ok, "outer_exit": 3}, "VOID",
              regs, (base, base + n)),
             ("survivors: refused", upto(5), {**ok, "outer_exit": 137, "survivors": "123 "}, None, regs, "FAULT"),
             ("slots out of order: refused", st + [g(1), g(0)], {**ok, "outer_exit": 124}, None, regs, "FAULT"),
             ("a slot started twice: refused", upto(2) + [g(1)], {**ok, "outer_exit": 3}, "VOID", regs, "FAULT"),
             (f"more than {n} starts: refused", upto(n + 1), {**ok, "outer_exit": 3}, "VOID", regs, "FAULT"),
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
    verdict = json.loads((ROOT / RUN / "record.json").read_text()).get("status") if len(inputs) > 3 else None
    regs = {n: [tuple(iv) for iv in getattr(REG, f"{n}_SEED_INTERVALS")] for n in ("ACCOUNTED", "EXPOSED", "RETIRED")}
    faults, expose, retire = reconcile(plan, events, outer, verdict, regs)
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    lines = ["H-TB PILOT -- CONFIRMATION-TEST SEED-USAGE DERIVATION (closing)", "=" * 96,
             f"HEAD {_git('rev-parse', 'HEAD').stdout.strip()}; working tree {'DIRTY' if dirty else 'clean'}",
             *(f"input {rel} (committed, unmodified)" for rel in inputs),
             f"outer exit {outer.get('outer_exit')}; survivors {outer.get('survivors')!r}; status {verdict!r}",
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
