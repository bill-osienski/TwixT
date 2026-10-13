"""SEARCH DISTILLATION STAGE B -- THE 441-SLOT SEED SURVEY AND ITS COLLISION PROOFS (read-only).

Card `docs/superpowers/2026-10-12-search-distillation-stage-b-card.md` §9: one fresh block of 441 slots --
200 corpus games, 200 label seeds, 40 repeat seeds and the training schedule seed.

A COPY of the Stage A survey (`evidence/search-distillation-stage-a-seed-survey/01_seed_survey.py`, run at
6c7177c1/94b32ed1/a809308f); its diff from that file is limited to: the block size (441), the paths, the slot
lines for the card's four roles, the plan's kind, the titles, and the most recent EXPOSED block's label.

THE METHOD IS IMPORTED, NOT COPIED: the tactical line's survey and collision-proof scripts, each pinned by
sha256 -- the union scanner, the prior terms (four registries, CONSUMED_SEEDS, every named spent block, H3's
generation ranges, the four XOR masks), the check (direct overlap, derived streams, injectivity, gap floor =
the block's own size), the negative, differential and exemption controls, and the FAIL-CLOSED enumeration.

THE RULE: `seed_selection.candidate(reserved, 441)` (ca9cb1eb): BASE = the least multiple of 100 leaving at
least 441 unused seeds above the highest registered or named-spent seed in the 2026 band; block = [BASE,
BASE + 441). Slot s has seed BASE + s: G_g (game g) slots 0-199, L_g (labels of game g) 200-399, R_h (repeats
of held-out game 5h + 4) 400-439, T (the training schedule) 440. Checked, never adjusted.

THE SCOPE is the pinned scope in TWO trees: this clean linked worktree and the main checkout (the parent of
the shared git directory), every regular file under the three roots of both, fail-closed. Only THIS tree
must be clean.

THREE MODES, each run ONCE on a clean committed tree, each writing ONE create-only record here:
  survey -> 02_seed_survey_run.txt
  pre    -> 03_collision_proof_pre_run.txt   (survey record committed) registry unchanged, block
                                              unregistered, no plan
  post   -> 04_collision_proof_post_run.txt  (after a SEPARATELY AUTHORIZED registration) ACCOUNTED = the
                                              survey's + exactly the block appended last, nothing else
                                              changed, the plan committed and equal to build_plan
Exit 0 proven / 1 not proven / 2 stopped (no verdict) / 4 refused. Nothing is drawn, registered, exposed
or consumed. This script holds no seed literal.

    python -B <this file> --mode survey|pre|post
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                    # the repository, before any of its modules is imported
from scripts.GPU.alphazero.search_distillation_stage_b import GAME, LABEL, REPEAT, TRAIN, N_SLOTS  # 0, 200, 400, 440, 441  # noqa: E402
HERE = "docs/superpowers/evidence/search-distillation-stage-b-seed-survey"
PLAN = "docs/superpowers/evidence/search-distillation-stage-b/plan.json"
OUT = {"survey": f"{HERE}/02_seed_survey_run.txt", "pre": f"{HERE}/03_collision_proof_pre_run.txt",
       "post": f"{HERE}/04_collision_proof_post_run.txt"}
OWN_RECORDS = (f"{HERE}/01_seed_survey.py", *OUT.values())                     # [3] is the post record
#: the method, pinned (the same two files the tactical line and stages 1-3 pinned)
PINNED = {"survey": ("docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py",
                     "6f6e55dbd5da672848c757753d4e61a3e9d110d96569b7ef468f902cf46f8860"),
          "proof": ("docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py",
                    "dd192011938ca5c6674092b243681174e669e1e96b22b8d13da2a264b340764d")}
VERDICT_OK = "THE STAGE-B BLOCK IS CLEAN"


class Refused(Exception):
    """A pin, the record or the mode is not what it must be: nothing is scanned or written."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_method(root=ROOT):
    """The two pinned modules, private copies; this line's values rebound on the SURVEY copy only."""
    mods = {}
    for key, (rel, want) in PINNED.items():
        p = root / rel
        if not p.is_file() or _sha(p.read_bytes()) != want:
            raise Refused(f"{rel} is not the pinned file")
        spec = importlib.util.spec_from_file_location(f"_search_distillation_stage_b_{key}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[key] = mod
    S = mods["survey"]
    S.N, S.HERE, S.PROOF, S.PLAN, S.OWN_RECORDS = N_SLOTS, HERE, HERE, PLAN, OWN_RECORDS
    from scripts.GPU.alphazero import seed_selection as SEL
    return S, mods["proof"], SEL


def build_plan(base, survey_sha):
    """The frozen plan: BASE and the slot count only (never the exclusive end), bound to the survey record."""
    return {"kind": "search-distillation-stage-b-plan/1", "base": base, "n_slots": N_SLOTS,
            "survey_record_sha256": survey_sha}


def slot_lines(block):
    """Card §9: G_g corpus game g, L_g the labels of game g, R_h the repeats of held-out game 5h + 4, T the schedule."""
    role = lambda s: (f"G game {s - GAME}" if s < LABEL else f"L game {s - LABEL}" if s < REPEAT else
                      f"R game {5 * (s - REPEAT) + 4}" if s < TRAIN else "T schedule")
    return [f"    SLOT {s} {role(s)} seed {block[0] + s}" for s in range(N_SLOTS)]


def read_record(text, S, SEL):
    """The survey record's facts -- refused unless CLEAN, every control passed, a clean tree, and its
    block, BASE and slots are the rule's."""
    def one(pattern, what):
        found = re.findall(pattern, text, re.M)
        if len(found) != 1:
            raise Refused(f"the survey record does not state {what} exactly once")
        return found[0]
    if one(r"^SURVEY VERDICT   : (.*)$", "its verdict") != VERDICT_OK:
        raise Refused("the survey record's verdict is not CLEAN")
    passed, total = one(r"^controls passed: (\d+)/(\d+)$", "its controls")
    if passed != total:
        raise Refused("the survey record has a failed control")
    head, state = one(r"^HEAD ([0-9a-f]{40}); working tree (\w+)$", "its HEAD")
    if state != "clean":
        raise Refused("the survey ran on a dirty tree")
    registry_sha = one(rf"^registry {re.escape(S.REGISTRY_SOURCE)} sha256 ([0-9a-f]{{64}})$", "the registry hash")
    base, lo, hi = (int(x) for x in one(r"^CANDIDATE RULE: .*; BASE = (\d+); block = \[(\d+), (\d+)\) -- ", "BASE"))
    if one(r"^THE BLOCK \((\d+, \d+)\) .* CLEAN=(\w+)$", "the block's check") != (f"{lo}, {hi}", "True"):
        raise Refused("the survey record's block check is not CLEAN, or not this block")
    block = (lo, hi)
    if block != (base, base + N_SLOTS) or base % SEL.ALIGN or not SEL.BAND[0] <= lo < hi <= SEL.BAND[1] or \
            re.findall(r"^    SLOT .*$", text, re.M) != slot_lines(block):
        raise Refused("the survey record's block, BASE and slots are not the rule's")
    return {"head": head, "registry_sha256": registry_sha, "base": base, "block": block}


def state_faults(mode, record, cur_text, pre_text, plan, plan_tracked, survey_sha, S, P):
    """pre: the tactical proof's own pre-mode check. post: ACCOUNTED = the survey's + exactly the block
    appended last, every other registry and statement unchanged, the plan committed and = build_plan."""
    if mode == "pre":
        return P.state_faults("pre", record, cur_text, None, plan, plan_tracked, S)
    block, faults = record["block"], []
    if _sha(pre_text.encode()) != record["registry_sha256"]:
        faults.append("the pre-registration registry is not the one the survey read")
    cur, cur_others = P._assignments(cur_text, S)
    pre, pre_others = P._assignments(pre_text, S)
    if cur["ACCOUNTED_SEED_INTERVALS"] != pre["ACCOUNTED_SEED_INTERVALS"] + [block]:
        faults.append("ACCOUNTED is not the survey's state with exactly the block appended last")
    for name in S.REGISTRY_NAMES[1:]:
        if cur[name] != pre[name]:
            faults.append(f"{name} changed")
    if cur_others != pre_others:
        faults.append("a statement other than ACCOUNTED changed in the registry")
    if plan is None or not plan_tracked:
        faults.append("the Stage B plan is absent or not committed")
    elif plan != build_plan(block[0], survey_sha):
        faults.append("the Stage B plan is not build_plan of the surveyed block")
    return faults


_LINES = []


def say(*a):
    print(*a, flush=True)
    _LINES.append(" ".join(str(x) for x in a))


def _state_controls(S, P, record, survey_sha):
    """Synthetic registries and plans: the expected state must hold for the right ones only."""
    lo, hi = record["block"]

    def reg(accounted, exposed="()", other="X = 1"):
        return (f"ACCOUNTED_SEED_INTERVALS = (\n{accounted})\nEXPOSED_SEED_INTERVALS = {exposed}\n"
                f"RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n{other}\n")
    old = f"    ({lo - 9000}, {lo - 8000}),\n"
    pre = reg(old)
    rec = {**record, "registry_sha256": _sha(pre.encode())}
    plan = build_plan(lo, survey_sha)
    good = reg(old + f"    ({lo}, {hi}),\n")
    wrong_sha = ("1" if survey_sha[0] == "0" else "0") + survey_sha[1:]   # differs from survey_sha in every mode
    cases = [("pre: the survey's registry, no plan", "pre", pre, None, None, True),
             ("pre: the registry changed", "pre", reg(old, other="X = 2"), None, None, False),
             ("pre: the block already registered", "pre", good, None, None, False),
             ("pre: a plan already exists", "pre", pre, None, plan, False),
             ("post: exactly the block appended last, the plan", "post", good, pre, plan, True),
             ("post: the block appended FIRST", "post", reg(f"    ({lo}, {hi}),\n" + old), pre, plan, False),
             ("post: a wider tuple", "post", reg(old + f"    ({lo}, {hi + 1}),\n"), pre, plan, False),
             ("post: EXPOSED changed too", "post", reg(old + f"    ({lo}, {hi}),\n", exposed=f"(({lo}, {hi}),)"),
              pre, plan, False),
             ("post: no plan", "post", good, pre, None, False),
             ("post: a plan with another BASE", "post", good, pre, {**plan, "base": lo + 1}, False),
             (f"post: a plan of {N_SLOTS + 1} slots", "post", good, pre, {**plan, "n_slots": N_SLOTS + 1}, False),
             ("post: a plan bound to another survey record", "post", good, pre,
              {**plan, "survey_record_sha256": wrong_sha}, False)]
    out = []
    for label, mode, cur, pre_text, p, want_ok in cases:
        faults = state_faults(mode, rec, cur, pre_text, p, p is not None, survey_sha, S, P)
        out.append((label, (not faults) == want_ok, "; ".join(faults) or "the expected state"))
    return out


def _git(*args, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()


def main_checkout():
    """The main checkout: the parent of the shared git directory (== ROOT when not run in a linked worktree)."""
    common = pathlib.Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent.resolve()


def main(argv):
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else None
    if mode not in OUT:
        raise Refused("--mode must be survey, pre or post")
    out = ROOT / OUT[mode]
    if out.exists():
        raise Refused(f"{OUT[mode]} exists: each mode runs once (create-only)")
    S, P, SEL = load_method()
    main_root = main_checkout()
    head, dirty = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    say(f"SEARCH DISTILLATION STAGE B -- SEED {'SURVEY' if mode == 'survey' else 'COLLISION PROOF, mode ' + mode.upper()}")
    say("=" * 96)
    say(f"HEAD {head}; working tree {'DIRTY' if dirty else 'clean'}")
    say(f"this tree {ROOT}")
    if main_root != ROOT.resolve():
        n_dirty = len(_git("status", "--porcelain", root=main_root).splitlines())
        say(f"main checkout {main_root} (scanned too; HEAD {_git('rev-parse', 'HEAD', root=main_root)}; "
            f"{n_dirty} uncommitted entries, not required clean)")
    for key, (rel, sha) in PINNED.items():
        say(f"method ({key}) {rel} sha256 {sha} (pinned)")
    sel_rel = "scripts/GPU/alphazero/seed_selection.py"
    say(f"rule {sel_rel} sha256 {_sha((ROOT / sel_rel).read_bytes())} (ALIGN {SEL.ALIGN})")
    cur_text = (ROOT / S.REGISTRY_SOURCE).read_text(encoding="utf-8")
    say(f"registry {S.REGISTRY_SOURCE} sha256 {_sha(cur_text.encode())}")
    registries, named, gen, masks, consumed = S.prior_terms()
    say(f"masks (imported): {[hex(m) for m in masks]}")
    say("registries: " + ", ".join(f"{k} {len(v)}" for k, v in registries.items()))
    say("named spent blocks: " + ", ".join(f"{k} {len(v)}" for k, v in named.items()) +
        f"; H3 generation ranges {len(gen)}; CONSUMED_SEEDS {len(consumed)}")
    say(f"EXEMPT PATHS (exact; plus registry tuple elements by parsing): {len(OWN_RECORDS)} + the plan")
    for p in (*OWN_RECORDS, f"{PLAN}  (only if every band literal in it is a slot of the block)"):
        say(f"    {p}")
    faults, survey_sha = [], None
    if mode == "survey":
        reserved = [iv for v in registries.values() for iv in v] + [iv for v in named.values() for iv in v]
        highest, base, block = SEL.candidate(reserved, N_SLOTS)
        say(f"\nCANDIDATE RULE: highest registered/named-spent SEED in the band = {highest}; BASE = {base}; "
            f"block = [{block[0]}, {block[1]}) -- {N_SLOTS} seeds")
    else:
        rel = OUT["survey"]
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", rel],
                                 capture_output=True).returncode == 0
        if not tracked or subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode:
            raise Refused("the survey record is absent, uncommitted or modified")
        raw = (ROOT / rel).read_bytes()
        survey_sha = _sha(raw)
        record = read_record(raw.decode("utf-8"), S, SEL)
        block = record["block"]
        say(f"\nsurvey record {rel} sha256 {survey_sha} (read at {record['head']})")
        say(f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {record['base']}")
        pre_text = cur_text if mode == "pre" else subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{record['head']}:{S.REGISTRY_SOURCE}"],
            capture_output=True, text=True, check=True).stdout
        plan_path = ROOT / PLAN
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
        plan_tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", PLAN],
                                      capture_output=True).returncode == 0
        faults = state_faults(mode, record, cur_text, pre_text, plan, plan_tracked, survey_sha, S, P)
        say(f"EXPECTED STATE ({mode}): {'HOLDS' if not faults else 'FAILS -- ' + '; '.join(faults)}")
        registries = P.prior_registries(mode, registries, block)
    try:
        files, links = P.strict_files(S)
        literals = S.repo_literals(files, block)
        extra, extra_links = [], []
        if main_root != ROOT.resolve():
            extra, extra_links = P.strict_files(S, main_root)
            for v, where in S.repo_literals(extra, block, main_root).items():
                literals.setdefault(v, []).extend((f"<main>/{rel}", n) for rel, n in where)
    except S.SurveyStop as e:
        say(f"\nSTOPPED (fail-closed): {e} -- NO VERDICT")
        _write(out)
        return 2
    say(f"\nSCOPE (fail-closed enumeration) under {list(S.ROOTS)}: this tree {len(files)} regular files, "
        f"{len(links)} symbolic links (not followed); main checkout {len(extra)} regular files, "
        f"{len(extra_links)} symbolic links; band literal values counted: {len(literals)}")
    prior = (registries, named, gen, consumed, literals)
    r = S.check(block, prior, masks)
    say(f"THE BLOCK {r['block']}  n={r['n']}  direct={r['direct'] or 'NONE'}  streams={r['streams']}  "
        f"injective={r['injective']}  gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    say("SLOTS (card §9: G 0-199, L 200-399, R 400-439, T 440; slot s has seed BASE + s):")
    for line in slot_lines(block):
        say(line)
    say("\nCONTROLS")
    say("-" * 96)
    negatives = [(f"{k} block {iv}", iv) for k, ivs in named.items() for iv in ivs[:1]]
    negatives += [("the TEST_ONLY band", registries["TEST_ONLY"][0]),
                  ("inside the live H3 uniform generation range", (gen[-1][0], gen[-1][0] + N_SLOTS)),
                  ("overlaps the block by one seed", (block[1] - 1, block[1] + N_SLOTS - 1)),
                  ("starts where the block ends -- inside the floor", (block[1], block[1] + N_SLOTS)),
                  ("one seed inside the floor", (block[1] + N_SLOTS - 1, block[1] + 2 * N_SLOTS - 1)),
                  ("the most recently EXPOSED block (Stage A's games)", registries["EXPOSED"][-1])]
    mutation = S.mutation_literal()
    negatives.append((f"a block at the H4-4b MUTATION literal {mutation}", (mutation, mutation + N_SLOTS)))
    results = [(f"REJECTED: {why}", not S.check(b, prior, masks, others=(block,))["clean"], "") for why, b in negatives]
    results += S._plant_controls(block, prior, masks, literals, r["clean"])
    results += S._exemption_controls(block)
    results += P._enumeration_controls(S)
    probe_rec = {"head": head, "registry_sha256": _sha(cur_text.encode()), "base": block[0], "block": block}
    results += _state_controls(S, P, probe_rec, survey_sha or "0" * 64)
    passed = 0
    for label, ok, detail in results:
        passed += ok
        say(f"  {'PASS' if ok else 'FAIL':6s} {label}{'  [' + detail + ']' if detail else ''}")
    say(f"\ncontrols passed: {passed}/{len(results)}")
    ok = r["clean"] and not faults and passed == len(results) and not dirty
    if mode == "survey":
        say(f"SURVEY VERDICT   : {VERDICT_OK if r['clean'] else 'NOT CLEAN -- STOP (no adjustment)'}"
            f"{'' if passed == len(results) else ' -- A CONTROL FAILED'}{' -- RUN ON A DIRTY TREE' if dirty else ''}")
    else:
        say(f"PROOF VERDICT ({mode}): {'THE STAGE-B BLOCK IS CLEAN AND THE STATE IS AS EXPECTED' if ok else 'NOT PROVEN -- STOP'}"
            f"{'' if r['clean'] else ' -- NOT CLEAN'}{' -- STATE' if faults else ''}"
            f"{'' if passed == len(results) else ' -- A CONTROL FAILED'}{' -- DIRTY TREE' if dirty else ''}")
    _write(out)
    return 0 if ok else 1


def _write(out):
    with open(out, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    try:
        sys.exit(main(sys.argv))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
