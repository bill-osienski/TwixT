"""THE COMBINATION POLICY PILOT -- THE 64-SLOT SCREEN GAME-SEED SURVEY AND ITS COLLISION PROOFS, IN THE 2027 BAND
(read-only).

Card docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 step 5 (`combination_policy_pilot run screen`: 32
colour-reversed pairs; game g plays seed BASE + g through eval_runner.build_agent_pairing_tasks). BUILT ONLY (ledger,
2026-10-31, the user's review): running any mode is a separate approval. A copy of
evidence/combination-policy-generate-seed-survey/01_seed_survey.py (generation's, whose description follows); the diff:
the block size (64), the paths, the plan's kind, the module name, the titles and verdict lines, the slot listing's
heading (the screen's games, played in order in one process), and ONE CONTROL'S EXPECTED VALUE: the two-band rule on an
empty band gives the least multiple of 100 at least 64 above the band's edge, which for 64 is not the edge + 64 that
the copied control wrote out for 1,000 (a multiple of 100); the control now states the rule itself, and the rule
(`seed_band_2027.candidate`) is unchanged. Read the copied description's 1000 as 64 and its corpus games as the
screen's games. A block the check finds NOT CLEAN is returned for review, never adjusted.

THE COMBINATION POLICY PILOT -- THE 1,000-SLOT CORPUS-GENERATION GAME-SEED SURVEY AND ITS COLLISION PROOFS, IN THE
2027 BAND (read-only).

Amendment docs/superpowers/2026-10-30-combination-policy-generation-amendment.md (`run generate`, card §5 step 3: 1,000
games of the proven agent playing itself; slot g plays seed BASE + g). BUILT ONLY (ledger, 2026-10-30): running any mode
is a separate approval. A copy of evidence/combination-policy-throughput2-seed-survey/01_seed_survey.py (whose
description follows); the diff: the block size (1000), the paths, the plan's kind, the module names, the titles and
verdict lines, and THE TWO-BAND METHOD (scripts/GPU/alphazero/seed_band_2027.py, the user's review 2026-10-30): the
2026 band's room ends at an H4 fixture literal, below which no 1,000-seed block fits, so
  * the pinned survey module's scanner is rebound (`seed_band_2027.install`) to see nine-digit literals of BOTH the
    2026 and the 2027 band, plain or underscore-grouped; the scope, exemptions, prior terms, four derived streams and
    the check (gap floor = the block's own size, 1000) are the pinned ones, unchanged;
  * the rule is `seed_band_2027.candidate`: BASE = the least multiple of 100 leaving at least 1000 seeds above the
    highest registered/named-spent seed AND the highest repository literal in the 2027 band (an empty band counts its
    lower edge); so in survey mode the scan runs first, once with no block (the plan is never exempt for the rule), and
    the rule reads those literals; the check then scans as before;
  * two-band controls join the copied ones: both prefixes, plain and grouped, are seen; other prefixes are not; the
    install took effect; the rule on an empty band and above the band's literals; a cross-band derived-stream
    collision (a synthetic mask) is rejected; with the real masks no 2026-band value shares a derived stream with a
    2027-band value.
Read the copied description's 144 as 1000 and its throughput games as the corpus games. A block the check finds NOT
CLEAN is returned for review, never adjusted.

THE COMBINATION POLICY PILOT -- THE 144-SLOT THROUGHPUT-CHECK ATTEMPT-2 GAME-SEED SURVEY AND ITS COLLISION PROOFS
(read-only).

Amendment docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md: the ONE fresh attempt (`run
throughput2`) needs a fresh block. BUILT ONLY (ledger, 2026-10-29): running any mode is a separate approval. A copy of
evidence/combination-policy-throughput-seed-survey/01_seed_survey.py (attempt 1's, whose description follows); the
diff: the paths, the plan's kind, the module names, the titles and verdict lines. Attempt 1's block is EXPOSED and
RETIRED whole, so the rule cannot select it again.

THE COMBINATION POLICY PILOT -- THE 144-SLOT THROUGHPUT-CHECK GAME-SEED SURVEY AND ITS COLLISION PROOFS (read-only).

Card docs/superpowers/2026-10-28-combination-policy-pilot-card.md §5 (step 2, the throughput check: 144 games of the
proven agent playing itself, `combination_policy_pilot run throughput`; slot g plays seed BASE + g, whichever worker
claims it -- combination_selfplay.generate -- through mirror_halving_combination.play_game, whose four streams
seed ^ 0xA5A5A5 / 0x5A5A5A / 0xC3C3C3 / 0x3C3C3C are exactly the method's four masks). Its own registered block,
"by the existing procedure". BUILT ONLY (ledger, decision note after entry 40): running any mode is a separate
approval. A copy of evidence/mirror-halving-strength2-seed-survey/01_seed_survey.py (attempt 2's, whose description
follows); the diff: the block size (144), the paths, the plan's kind, the module names, the titles and verdict lines,
and the slot lines (no pairs: `SLOT g seed BASE + g`). The method, the rule, the scope, the controls and the three
modes are unchanged. Read the copied description's 1018 as 144 and its games as this check's games.

MIRROR AVERAGING + ROOT SEQUENTIAL HALVING -- THE 1018-SLOT STRENGTH ATTEMPT-2 GAME-SEED SURVEY AND ITS COLLISION
PROOFS (read-only).

Amendment docs/superpowers/2026-10-27-mirror-halving-strength-attempt2-amendment.md: the ONE fresh attempt (`run
strength2`) needs a fresh block. A copy of evidence/mirror-halving-strength-seed-survey/01_seed_survey.py (attempt
1's, whose description follows); the diff: the paths, the plan's kind, the module names, the titles and verdict lines.
Attempt 1's block [202693600, 202694618) is EXPOSED in part and RETIRED whole, so the rule cannot select it again.

Card docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md §5 (T: 509 fresh colour-reversed
pairs, 1018 games of scripts/GPU/alphazero/mirror_halving_match.py `run strength`; game g has seed BASE + g from
eval_runner.build_agent_pairing_tasks, pair g // 2). A copy of
evidence/search-distillation-d2-heldout-attempt2-seed-survey/01_seed_survey.py (attempt 2's, run 2026-10-25); the
diff: the block size (1018), the paths, the plan's kind, the titles and verdict lines, the slot lines and the most
recent EXPOSED block's label. BUILT ONLY: running any mode is a separate approval. The copied description follows
(read its 12 as 1018, its held-out games as this run's games); attempt 1's description follows within it.

One fresh block of 12 game seeds for scripts/GPU/alphazero/search_distillation_d2_heldout.py (card
docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md, §4 and §6): 12 self-play games of
calib020_0001 at 1,600 simulations in ONE phase, all held out -- the trial's fresh held-out set. Game g:
random.Random(BASE + g), train()'s start draw.

A COPY of D2 Stage 2's survey (`evidence/search-distillation-d2-generate-seed-survey/01_seed_survey.py`, run at
88209d76/0d77665b/e10987bc); its diff from that file is limited to: the block size (12), the paths, the slot lines,
the plan's kind, the titles and verdict line, and the most recent EXPOSED block's label.

THE METHOD IS IMPORTED, NOT COPIED: the tactical line's survey and collision-proof scripts, each pinned by
sha256 -- the union scanner, the prior terms (four registries, CONSUMED_SEEDS, every named spent block, H3's
generation ranges, the four XOR masks), the check (direct overlap, derived streams, injectivity, gap floor =
the block's own size), the negative, differential and exemption controls, and the FAIL-CLOSED enumeration.

THE RULE: `seed_selection.candidate(reserved, 12)` (ca9cb1eb): BASE = the least multiple of 100 leaving at
least 12 unused seeds above the highest registered or named-spent seed in the 2026 band; block = [BASE,
BASE + 12). Slot g = game g has seed BASE + g, all in phase 0, all held out. Checked, never adjusted.

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
N_SLOTS = 64
HERE = "docs/superpowers/evidence/combination-policy-screen-seed-survey"
PLAN = "docs/superpowers/evidence/combination-policy-screen-run/plan.json"
OUT = {"survey": f"{HERE}/02_seed_survey_run.txt", "pre": f"{HERE}/03_collision_proof_pre_run.txt",
       "post": f"{HERE}/04_collision_proof_post_run.txt"}
OWN_RECORDS = (f"{HERE}/01_seed_survey.py", *OUT.values())                     # [3] is the post record
#: the method, pinned (the same two files the tactical line and stages 1-3 pinned)
PINNED = {"survey": ("docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py",
                     "6f6e55dbd5da672848c757753d4e61a3e9d110d96569b7ef468f902cf46f8860"),
          "proof": ("docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py",
                    "dd192011938ca5c6674092b243681174e669e1e96b22b8d13da2a264b340764d")}
VERDICT_OK = "THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN"


class Refused(Exception):
    """A pin, the record or the mode is not what it must be: nothing is scanned or written."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_method(root=ROOT):
    """The two pinned modules, private copies; this line's values rebound on the SURVEY copy only, its scanner
    extended to both bands; the rule is the two-band method's."""
    mods = {}
    for key, (rel, want) in PINNED.items():
        p = root / rel
        if not p.is_file() or _sha(p.read_bytes()) != want:
            raise Refused(f"{rel} is not the pinned file")
        spec = importlib.util.spec_from_file_location(f"_combination_policy_screen_{key}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[key] = mod
    S = mods["survey"]
    S.N, S.HERE, S.PROOF, S.PLAN, S.OWN_RECORDS = N_SLOTS, HERE, HERE, PLAN, OWN_RECORDS
    from scripts.GPU.alphazero import seed_band_2027 as SEL
    SEL.install(S)
    return S, mods["proof"], SEL


def build_plan(base, survey_sha):
    """The frozen plan: BASE and the slot count only (never the exclusive end), bound to the survey record."""
    return {"kind": "combination-policy-screen-plan/1", "base": base, "n_slots": N_SLOTS,
            "survey_record_sha256": survey_sha}


def slot_lines(block):
    return [f"    SLOT {g} seed {block[0] + g}"
            for g in range(N_SLOTS)]


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
        faults.append("the screen plan is absent or not committed")
    elif plan != build_plan(block[0], survey_sha):
        faults.append("the screen plan is not build_plan of the surveyed block")
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


def _two_band_controls(S, SEL, block, prior, masks):
    """The two-band method on synthetic values computed from the bands and the block (no literal is typed here)."""
    lo, hi = block
    old, new = SEL.BAND_2026[0] + 54_321, lo + 7
    out = []
    for label, text, want in (("a PLAIN 2026-band literal is seen", f"x = {old}\n", {old: 1}),
                              ("an UNDERSCORE-GROUPED 2026-band literal is seen", f"x = {old:_}\n", {old: 1}),
                              ("a PLAIN 2027-band literal is seen", f"x = {new}\n", {new: 1}),
                              ("an UNDERSCORE-GROUPED 2027-band literal is seen", f"x = {new:_}\n", {new: 1}),
                              ("neither band's neighbours are counted (the 2025 and 2028 prefixes, ten digits, a decimal)",
                               f"a = {SEL.BAND_2026[0] - 1}\nb = {SEL.BAND[1]}\nc = {new}0\nd = 0.{new}\n", {})):
        got = S.scan(text)
        out.append((label, got == want, str(got)))
    out.append(("the install took effect: the pinned plain pattern alone misses a 2027-band literal",
                S.scan is SEL.scan and not S.V16.findall(f"x = {new}\n"), ""))
    e = SEL.BAND[0]
    b0 = next(b for b in range(e + N_SLOTS, e + N_SLOTS + SEL.ALIGN) if b % SEL.ALIGN == 0)
    out.append(("the rule on an EMPTY band: the least multiple of 100 at least N above the band's edge",
                SEL.candidate([], [], N_SLOTS) == (None, b0, (b0, b0 + N_SLOTS)), ""))
    got = SEL.candidate([(e + 10, e + 20)], [e + 5_000], N_SLOTS)
    out.append(("the rule lies above the band's highest literal as well as its intervals",
                got[0] == e + 5_000 and got[2][0] - (e + 5_001) >= N_SLOTS and got[2][0] % SEL.ALIGN == 0, str(got)))
    registries, named, gen, consumed, literals = prior
    for k in range(50):                            # a 2026-band value the prior terms leave untouched under its mask
        v = SEL.BAND_2026[0] + 54_321 + 1_009 * k
        m = lo ^ v                                 # a synthetic mask taking the block's first seed onto it
        r0 = S.check(block, prior, [m])
        if r0["streams"] == 0:
            break
    r = S.check(block, (registries, named, gen, consumed, {**literals, v: [("<synthetic>", 1)]}), [m])
    out.append(("REJECTED: a cross-band derived-stream collision (a synthetic mask onto a planted 2026-band literal)",
                r0["streams"] == 0 and r["streams"] >= 1 and not r["clean"],
                f"streams without the plant {r0['streams']}, with it {r['streams']}"))

    def der(rng):
        return {x for s in rng for x in (s, *(s ^ k for k in masks))}
    out.append(("with the real masks, no 2026-band value shares a derived stream with a 2027-band value",
                not der(range(*SEL.BAND_2026)) & der(range(*SEL.BAND)), f"masks {[hex(k) for k in masks]}"))
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
    say(f"COMBINATION POLICY PILOT -- SCREEN GAME-SEED {'SURVEY' if mode == 'survey' else 'COLLISION PROOF, mode ' + mode.upper()}")
    say("=" * 96)
    say(f"HEAD {head}; working tree {'DIRTY' if dirty else 'clean'}")
    say(f"this tree {ROOT}")
    if main_root != ROOT.resolve():
        n_dirty = len(_git("status", "--porcelain", root=main_root).splitlines())
        say(f"main checkout {main_root} (scanned too; HEAD {_git('rev-parse', 'HEAD', root=main_root)}; "
            f"{n_dirty} uncommitted entries, not required clean)")
    for key, (rel, sha) in PINNED.items():
        say(f"method ({key}) {rel} sha256 {sha} (pinned)")
    sel_rel = "scripts/GPU/alphazero/seed_band_2027.py"
    say(f"rule and scanner {sel_rel} sha256 {_sha((ROOT / sel_rel).read_bytes())} (ALIGN {SEL.ALIGN}; the rule's "
        f"band {list(SEL.BAND)}; literals scanned in {list(SEL.BAND_2026)} and {list(SEL.BAND)})")
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
        block = None                              # the rule reads the scan's literals: chosen below, after the scan
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
        if block is None:                         # survey: the rule's literals, the plan never exempt (no block yet)
            seen = set(S.repo_literals(files, (0, 0)))
            if main_root != ROOT.resolve():
                seen |= set(S.repo_literals(P.strict_files(S, main_root)[0], (0, 0), main_root))
            reserved = [iv for v in registries.values() for iv in v] + [iv for v in named.values() for iv in v]
            highest, base, block = SEL.candidate(reserved, seen, N_SLOTS)
            say(f"\nCANDIDATE RULE: highest registered/named-spent seed or repository literal in the 2027 band = "
                f"{highest if highest is not None else 'none (an empty band: its lower edge)'}; BASE = {base}; "
                f"block = [{block[0]}, {block[1]}) -- {N_SLOTS} seeds")
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
    say("SLOTS (combination_policy_pilot run screen: game g plays seed BASE + g, in order, in one process):")
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
                  ("the most recently EXPOSED block (whichever block was closed most recently at survey time)", registries["EXPOSED"][-1])]
    mutation = S.mutation_literal()
    negatives.append((f"a block at the H4-4b MUTATION literal {mutation}", (mutation, mutation + N_SLOTS)))
    results = [(f"REJECTED: {why}", not S.check(b, prior, masks, others=(block,))["clean"], "") for why, b in negatives]
    results += S._plant_controls(block, prior, masks, literals, r["clean"])
    results += S._exemption_controls(block)
    results += P._enumeration_controls(S)
    probe_rec = {"head": head, "registry_sha256": _sha(cur_text.encode()), "base": block[0], "block": block}
    results += _state_controls(S, P, probe_rec, survey_sha or "0" * 64)
    results += _two_band_controls(S, SEL, block, prior, masks)
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
        say(f"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN AND THE STATE IS AS EXPECTED' if ok else 'NOT PROVEN -- STOP'}"
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
