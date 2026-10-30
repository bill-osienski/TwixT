"""THE COMBINATION POLICY PILOT -- CORPUS GENERATION'S SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

Amendment docs/superpowers/2026-10-30-combination-policy-generation-amendment.md (`run generate`, 1,000 slots on the
2027-band block). BUILT ONLY (ledger, 2026-10-30): running it is a separate approval. A copy of
evidence/combination-policy-throughput2-seed-closing/01_derive_usage.py (whose description follows); the diff: the
block size (1000), the paths and the title. The reviewed parallel-worker rule is unchanged: the started slots must
form the prefix {0..k-1} AS A SET with the ONE final `claims` event equal to k, or the WHOLE block is EXPOSED; the
whole block is RETIRED whatever happened. Read the copied description's 144 as 1000. A generation TIMEOUT (VOID) is
closed by this rule like any other run; whether to continue is the user's separate assessment.

THE COMBINATION POLICY PILOT -- THE THROUGHPUT CHECK ATTEMPT 2'S SEED-USAGE DERIVATION for closing (read-only,
outcome-blind).

Amendment docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md (`run throughput2`). BUILT
ONLY (ledger, 2026-10-29): running it is a separate approval. A copy of
evidence/combination-policy-throughput-seed-closing/01_derive_usage.py (attempt 1's, run once at 3b6df508, whose
description follows); the diff: the paths and the title.

THE COMBINATION POLICY PILOT -- THE THROUGHPUT CHECK'S SEED-USAGE DERIVATION for closing (read-only, outcome-blind).

Card docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 (step 2: 144 slots, `combination_policy_pilot run
throughput`). BUILT ONLY (ledger, decision note after entry 40): running it is a separate approval. A copy of
evidence/mirror-halving-strength2-seed-closing/01_derive_usage.py (attempt 2's, whose description follows); the diff:
the block size (144), the paths, the titles, and THE CARD'S ONE DIFF (§5), in `reconcile` and its controls:

  * 18 workers claim slots in parallel, so `game_start` events (field `slot`, logged as each worker's start notice
    arrives) come out of order. The started slots must form the prefix {0..k-1} AS A SET, each exactly once, and
    the ONE final `claims` event (the claim counter, logged after the last game) must equal k.
  * IF THE RECORD CANNOT ESTABLISH k -- no `claims` event (a run stopped, crashed or VOID before generation
    returned), more than one, a counter that is not k, a gap or a slot started twice -- THE WHOLE BLOCK IS EXPOSED.
    A started slot outside [0, 144) is a FAULT (a seed outside the block may have been played).
  * A REFUSED (4) run launched nothing: k = 0, and any `game_start` or `claims` is a FAULT. Outer exit 0 needs
    k = 144 established and status COMPLETED.
  * The whole block [BASE, BASE + 144) is RETIRED, whatever happened.
Everything else -- the inputs, the outcome-blind reading (event names, slots and the claim counter only; never a
move, winner, reason or score), the registry checks, the create-only record and the exit codes -- is unchanged.

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
HERE = "docs/superpowers/evidence/combination-policy-generate-seed-closing"
RUN = "docs/superpowers/evidence/combination-policy-generate"
RUN_DIR = "docs/superpowers/evidence/combination-policy-generate-run"
INPUTS = [f"{RUN_DIR}/plan.json", f"{RUN}/progress.jsonl", f"{RUN_DIR}/outer.json"]
OUT = f"{HERE}/02_derive_usage_run.txt"
N_SLOTS = 1000


def started_k(events):
    """k when the record establishes it (card §5), else None: the game_start slots, as a set, are {0..k-1}, each
    once, and the one claims event's counter is k."""
    slots = [e.get("slot") for e in events if e.get("event") == "game_start"]
    claims = [e.get("claims") for e in events if e.get("event") == "claims"]
    if len(claims) != 1 or type(claims[0]) is not int:
        return None
    k = claims[0]
    if len(slots) != k or set(slots) != set(range(k)):
        return None
    return k


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
    outside = [s for s in slots if type(s) is not int or not 0 <= s < N_SLOTS]
    if outside:
        faults.append(f"game_start slots {outside} are outside the {N_SLOTS} slots")
    n_claims = sum(e.get("event") == "claims" for e in events)
    k = started_k(events)
    status = outer.get("outer_exit")
    if status == 0 and (k != N_SLOTS or verdict != "COMPLETED"):
        faults.append(f"outer exit 0 but k {k} (started {len(slots)}, claims events {n_claims}), verdict {verdict!r}")
    if status == 4:
        if slots or n_claims:
            faults.append(f"a REFUSED run logged {len(slots)} game starts and {n_claims} claims events")
        k = 0
    acc, exp, ret = (regs[n] for n in ("ACCOUNTED", "EXPOSED", "RETIRED"))
    if acc.count(block) != 1:
        faults.append("the block is not ACCOUNTED exactly once")
    if any(lo < block[1] and block[0] < hi for lo, hi in exp + ret):
        faults.append("a seed of the block is already EXPOSED or RETIRED")
    if faults:
        return faults, None, None
    if k is None:
        return [], block, block                               # k not established: expose the whole block
    return [], ((base, base + k) if k else None), block


def _controls(base):
    """Synthetic inputs: the rule must give the fixed answer and refuse every wrong shape."""
    n = N_SLOTS
    plan = {"base": base, "n_slots": n}
    regs = {"ACCOUNTED": [(base, base + n)], "EXPOSED": [], "RETIRED": []}
    st = [{"event": "start"}]
    g = lambda s: {"event": "game_start", "slot": s, "worker": s % 18}
    c = lambda k: [{"event": "claims", "claims": k}]
    shuffled = lambda k: [g(s) for s in sorted(range(k), key=lambda s: (s * 7919) % (k or 1))]   # out of order
    ok = {"outer_exit": 0, "survivors": ""}
    whole = (base, base + n)
    cases = [(f"completed: {n} starts out of order, claims {n}: expose all, retire all", st + shuffled(n) + c(n), ok,
              "COMPLETED", regs, whole),
             ("REFUSED before any game: expose NONE, retire all", st, {**ok, "outer_exit": 4}, "REFUSED", regs, None),
             ("VOID before generation (no claims): k not established, expose all", st, {**ok, "outer_exit": 3},
              "VOID", regs, whole),
             ("timeout during loading (no claims): expose all", st, {**ok, "outer_exit": 124}, None, regs, whole),
             ("timeout mid-run, slots 0-5 started (no claims): expose all", st + shuffled(6), {**ok, "outer_exit": 124},
              None, regs, whole),
             ("VOID after generation, slots 0-5 out of order, claims 6: expose slots 0-5", st + shuffled(6) + c(6),
              {**ok, "outer_exit": 3}, "VOID", regs, (base, base + 6)),
             ("claims 7 but slots 0-5 started (a claimed slot never reported): expose all", st + shuffled(6) + c(7),
              {**ok, "outer_exit": 3}, "VOID", regs, whole),
             ("a gap: slots 0, 1, 3 started, claims 3: expose all", st + [g(0), g(1), g(3)] + c(3),
              {**ok, "outer_exit": 3}, "VOID", regs, whole),
             ("a slot started twice: expose all", st + [g(0), g(1), g(1)] + c(3), {**ok, "outer_exit": 3}, "VOID", regs,
              whole),
             ("two claims events: expose all", st + shuffled(6) + c(6) + c(6), {**ok, "outer_exit": 3}, "VOID", regs,
              whole),
             (f"kill in the last game, all {n} started (no claims): expose all", st + shuffled(n),
              {**ok, "outer_exit": 137}, None, regs, whole),
             ("survivors: refused", st + shuffled(5), {**ok, "outer_exit": 137, "survivors": "123 "}, None, regs,
              "FAULT"),
             ("unknown survivors (no pgid): refused", st, {**ok, "outer_exit": 124, "survivors": "unknown: no readable "
              "pgid.txt "}, None, regs, "FAULT"),
             ("progress not beginning with start: refused", shuffled(2) + c(2), {**ok, "outer_exit": 3}, "VOID", regs,
              "FAULT"),
             (f"a started slot outside the block ({n}): refused", st + shuffled(3) + [g(n)] + c(4),
              {**ok, "outer_exit": 3}, "VOID", regs, "FAULT"),
             (f"exit 0 with {n - 1} starts: refused", st + shuffled(n - 1) + c(n - 1), ok, "COMPLETED", regs, "FAULT"),
             (f"exit 0 with {n} starts but no claims: refused", st + shuffled(n), ok, "COMPLETED", regs, "FAULT"),
             ("REFUSED after a game start: refused", st + [g(0)], {**ok, "outer_exit": 4}, "REFUSED", regs, "FAULT"),
             ("REFUSED with a claims event: refused", st + c(0), {**ok, "outer_exit": 4}, "REFUSED", regs, "FAULT"),
             ("block not ACCOUNTED: refused", st, {**ok, "outer_exit": 4}, "REFUSED", {**regs, "ACCOUNTED": []},
              "FAULT"),
             ("block already RETIRED in part: refused", st, {**ok, "outer_exit": 4}, "REFUSED",
              {**regs, "RETIRED": [(base + n - 1, base + n)]}, "FAULT")]
    out = []
    for label, events, outer, verdict, r, want in cases:
        faults, expose, retire = reconcile(plan, events, outer, verdict, r)
        good = bool(faults) if want == "FAULT" else (not faults and expose == want and retire == whole)
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
    lines = ["COMBINATION POLICY PILOT -- GENERATION SEED-USAGE DERIVATION (closing)", "=" * 96,
             f"HEAD {_git('rev-parse', 'HEAD').stdout.strip()}; working tree {'DIRTY' if dirty else 'clean'}",
             *(f"input {rel} (committed, unmodified)" for rel in inputs),
             f"outer exit {outer.get('outer_exit')}; survivors {outer.get('survivors')!r}; status {verdict!r}",
             f"game starts (slots, as logged): {[e.get('slot') for e in events if e.get('event') == 'game_start']}",
             f"claims events: {[e.get('claims') for e in events if e.get('event') == 'claims']}"]
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
