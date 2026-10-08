"""TACTICAL CANDIDATE, A8.8 STEP 5 -- THE COLLISION PROOF of the pilot block, PRE and POST registration.

Card `docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §18 A8.1-A8.2. Each mode runs
ONCE, on a clean committed tree, and writes ONE create-only record at a path A8.2 exempts:
  pre  -- before the registration commit      -> 02_collision_proof_pre_run.txt
  post -- on the committed registration       -> 03_collision_proof_post_run.txt

PINNED, NEVER TYPED HERE:
  * the survey RECORD (`.../tactical-candidate-seed-survey/02_seed_survey_run.txt`, sha256 below): the
    block, BASE, the 16 pairs' seeds, the registry's hash and the commit it was read at are READ from it;
  * the survey SCRIPT (sha256 below): its scope, decoding, exemptions and collision check are IMPORTED
    unchanged -- A8.1's scope and A8.2's exemptions are the survey's, not copies.

ONE CHANGE, on review (2026-10-08): the ENUMERATION FAILS CLOSED. The survey's `os.walk` silently
skipped a directory it could not read. Here a directory-read error, a missing root, or an entry that is
neither a regular file, a directory nor a symbolic link STOPS the proof with no verdict. The historical
survey and its record are untouched.

THE EXPECTED REGISTRY STATE, checked separately by mode:
  pre  -- the registry is byte-identical to the one the survey read; no registry holds any seed of the
          block; no pilot plan exists.
  post -- the registry differs from that state ONLY by ONE tuple (BASE, BASE + 32) appended LAST to
          ACCOUNTED_SEED_INTERVALS (by parsing; comments are free); EXPOSED, RETIRED, TEST_ONLY and every
          other statement unchanged; the pilot plan committed, mode "pilot", its 16 pairs exactly the
          survey's ids and seeds. The block's own ACCOUNTED interval is no collision with itself: it, and
          only it, leaves the prior terms. (The plan's literals are exempt by A8.2's own condition.)
Then the survey's check -- direct overlap, derived streams, injectivity, the 32-seed floor -- and its
controls, plus the fail-closed enumeration and state controls. CLEAN, every control passed, the expected
state and a clean tree: exit 0. Otherwise 1; STOPPED: 2. No retry; nothing is adjusted. No seed literal
is written here.

    python <this file> --mode pre|post
"""
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = "docs/superpowers/evidence/tactical-candidate-seed-registration"
SURVEY_SCRIPT = "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py"
SURVEY_RECORD = "docs/superpowers/evidence/tactical-candidate-seed-survey/02_seed_survey_run.txt"
SURVEY_SCRIPT_SHA256 = "6f6e55dbd5da672848c757753d4e61a3e9d110d96569b7ef468f902cf46f8860"
SURVEY_RECORD_SHA256 = "dee36562d62bc4f449c07b841c859c562d9edb1dd42870bd260dcdea0ae91ee4"
OUT = {"pre": f"{HERE}/02_collision_proof_pre_run.txt", "post": f"{HERE}/03_collision_proof_post_run.txt"}


class ProofRefused(Exception):
    """A pin, the record or the mode is not what it must be: nothing is scanned or written."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


# ─────────────────────────────── the pins ───────────────────────────────

def load_survey(root=ROOT):
    """The pinned survey script, imported; refuses unless both pins hold."""
    for rel, want in ((SURVEY_SCRIPT, SURVEY_SCRIPT_SHA256), (SURVEY_RECORD, SURVEY_RECORD_SHA256)):
        p = root / rel
        if not p.is_file() or _sha(p.read_bytes()) != want:
            raise ProofRefused(f"{rel} is not the pinned file")
    spec = importlib.util.spec_from_file_location("_pilot_seed_survey", root / SURVEY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_record(text, S):
    """The survey record's facts -- refused unless it is a CLEAN verdict on a clean tree with every
    control passed, and its block, BASE and pairs agree with the survey's own rules."""
    def one(pattern, what):
        found = re.findall(pattern, text, re.M)
        if len(found) != 1:
            raise ProofRefused(f"the survey record does not state {what} exactly once")
        return found[0]
    if one(r"^SURVEY VERDICT   : (.*)$", "its verdict") != "THE PILOT BLOCK IS CLEAN":
        raise ProofRefused("the survey record's verdict is not CLEAN")
    passed, total = one(r"^controls passed: (\d+)/(\d+)$", "its controls")
    if passed != total:
        raise ProofRefused("the survey record has a failed control")
    head, state = one(r"^HEAD ([0-9a-f]{40}); working tree (\w+)$", "its HEAD")
    if state != "clean":
        raise ProofRefused("the survey ran on a dirty tree")
    registry_sha = one(rf"^registry {re.escape(S.REGISTRY_SOURCE)} sha256 ([0-9a-f]{{64}})$", "the registry hash")
    base, lo, hi = (int(x) for x in one(r"^CANDIDATE RULE: .*; BASE = (\d+); block = \[(\d+), (\d+)\) -- ", "BASE"))
    block = (lo, hi)
    if one(r"^THE BLOCK \((\d+, \d+)\) .* CLEAN=(\w+)$", "the block's check") != (f"{lo}, {hi}", "True"):
        raise ProofRefused("the survey record's block check is not CLEAN, or not this block")
    pairs = [{"pair_id": p, "red": int(r), "black": int(b)}
             for p, r, b in re.findall(r"^    (pilot-\d\d)  red (\d+)  black (\d+)$", text, re.M)]
    if block != (base, base + S.N) or base % S.ALIGN or not S.BAND[0] <= lo < hi <= S.BAND[1] or \
            pairs != S.pair_seeds(block):
        raise ProofRefused("the survey record's block, BASE and pairs are not the frozen rule's")
    return {"head": head, "registry_sha256": registry_sha, "base": base, "block": block, "pairs": pairs}


# ─────────────────────────────── the enumeration, FAIL-CLOSED ───────────────────────────────

def strict_files(S, root=ROOT):
    """The survey's scope (its ROOTS; every regular file, any suffix, any git state; symbolic links
    not followed), enumerated FAIL-CLOSED: a root that is not a directory, a directory that cannot be
    read, or an entry that is none of file, directory or symbolic link STOPS (S.SurveyStop)."""
    files, links = [], []

    def failed(e):
        raise S.SurveyStop(f"a directory could not be read ({type(e).__name__}: {e})")
    for base in S.ROOTS:
        top = root / base
        if top.is_symlink() or not top.is_dir():
            raise S.SurveyStop(f"{top}: the root is not a directory")
        for dirpath, dirnames, filenames in os.walk(top, onerror=failed, followlinks=False):
            for name in sorted(dirnames + filenames):
                p = pathlib.Path(dirpath) / name
                try:
                    mode = os.lstat(p).st_mode
                except OSError as e:
                    raise S.SurveyStop(f"{p}: cannot be examined ({type(e).__name__}: {e})") from None
                if stat.S_ISLNK(mode):
                    links.append(p)
                elif stat.S_ISREG(mode):
                    files.append(p)
                elif not stat.S_ISDIR(mode):
                    raise S.SurveyStop(f"{p}: neither a regular file, a directory nor a symbolic link")
    return sorted(files), sorted(links)


# ─────────────────────────────── the expected state, by mode ───────────────────────────────

def _assignments(text, S):
    """{registry name: [(lo, hi)]} by parsing, and the dumps of every OTHER top-level statement."""
    regs, others = {}, []
    for node in ast.parse(text).body:
        names = [getattr(t, "id", None) for t in getattr(node, "targets", [])]
        if isinstance(node, ast.Assign) and len(names) == 1 and names[0] in S.REGISTRY_NAMES:
            if names[0] in regs:
                raise ProofRefused(f"the registry assigns {names[0]} more than once")
            regs[names[0]] = [tuple(iv) for iv in ast.literal_eval(node.value)]
        else:
            others.append(ast.dump(node))
    if set(regs) != set(S.REGISTRY_NAMES):
        raise ProofRefused("the registry does not assign each of the four registries exactly once")
    return regs, others


def state_faults(mode, record, cur_text, pre_text, plan, plan_tracked, S):
    """The expected registry and plan state (A8.2) -- a list of faults, empty if it holds."""
    block, faults = record["block"], []
    cur, cur_others = _assignments(cur_text, S)
    if mode == "pre":
        if _sha(cur_text.encode()) != record["registry_sha256"]:
            faults.append("the registry is not the one the survey read")
        held = [(k, iv) for k, ivs in cur.items() for iv in ivs if iv[0] < block[1] and block[0] < iv[1]]
        if held:
            faults.append(f"a registry already holds seeds of the block: {held}")
        if plan is not None:
            faults.append("a pilot plan already exists")
        return faults
    if _sha(pre_text.encode()) != record["registry_sha256"]:
        faults.append("the pre-registration registry is not the one the survey read")
    pre, pre_others = _assignments(pre_text, S)
    if cur["ACCOUNTED_SEED_INTERVALS"] != pre["ACCOUNTED_SEED_INTERVALS"] + [block]:
        faults.append("ACCOUNTED is not the survey's state with exactly the block appended last")
    for name in S.REGISTRY_NAMES[1:]:
        if cur[name] != pre[name]:
            faults.append(f"{name} changed")
    if cur_others != pre_others:
        faults.append("a statement other than ACCOUNTED changed in the registry")
    if plan is None or not plan_tracked:
        faults.append("the pilot plan is absent or not committed")
    elif not isinstance(plan, dict) or plan.get("mode") != "pilot" or not isinstance(plan.get("pairs"), list) or \
            [(p.get("pair_id"), (p.get("seeds") or {}).get("red"), (p.get("seeds") or {}).get("black"))
             for p in plan["pairs"]] != [(p["pair_id"], p["red"], p["black"]) for p in record["pairs"]]:
        faults.append("the pilot plan's mode, pairs or seeds are not the survey's")
    return faults


def prior_registries(mode, registries, block):
    """Post: the block's own ACCOUNTED interval -- appended last, checked above -- is no collision with
    itself; it, and only it, leaves the prior terms."""
    if mode == "post" and registries["ACCOUNTED"] and registries["ACCOUNTED"][-1] == block:
        return {**registries, "ACCOUNTED": registries["ACCOUNTED"][:-1]}
    return registries


def verdict(clean, faults, passed, total, dirty):
    """Proven only if the block is CLEAN, the expected state holds, EVERY control passed and the tree
    is clean -- and the record says which failed."""
    ok = clean and not faults and passed == total and not dirty
    text = ("THE PILOT BLOCK IS CLEAN AND THE STATE IS AS EXPECTED" if ok else "NOT PROVEN -- STOP") + \
        ("" if clean else " -- NOT CLEAN") + (" -- STATE" if faults else "") + \
        ("" if passed == total else " -- A CONTROL FAILED") + (" -- DIRTY TREE" if dirty else "")
    return ok, text


# ─────────────────────────────── the run ───────────────────────────────

_LINES = []


def say(*a):
    print(*a, flush=True)
    _LINES.append(" ".join(str(x) for x in a))


def _enumeration_controls(S):
    """The fail-closed enumeration, on temporary trees: each case must STOP; a sound tree must not."""
    out = []
    cases = ("an UNREADABLE directory", "an UNREADABLE nested directory", "a missing root", "a root that is a file",
             "a FIFO inside a root", "a sound tree with a symbolic link")
    for label in cases:
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            for base in S.ROOTS:
                (t / base).mkdir(parents=True)
                (t / base / "ok.txt").write_text("x")
            locked = None
            if label == "an UNREADABLE directory":
                locked = t / "tests" / "locked"
            elif label == "an UNREADABLE nested directory":
                locked = t / "docs/superpowers/evidence/a/b"
            if locked is not None:
                locked.mkdir(parents=True)
                (locked / "hidden.txt").write_text("x")
                locked.chmod(0)
            if label == "a missing root":
                (t / "tests" / "ok.txt").unlink()
                (t / "tests").rmdir()
            if label == "a root that is a file":
                (t / "scripts" / "ok.txt").unlink()
                (t / "scripts").rmdir()
                (t / "scripts").write_text("not a directory")
            if label == "a FIFO inside a root":
                os.mkfifo(t / "scripts" / "pipe")
            if label == "a sound tree with a symbolic link":
                os.symlink(t / "tests" / "ok.txt", t / "tests" / "link.txt")
            try:
                files, links = strict_files(S, t)
                stopped, detail = False, f"{len(files)} files, {len(links)} links"
            except S.SurveyStop as e:
                stopped, detail = True, str(e).split(" (")[0].replace(str(t), "<tmp>")
            finally:
                if locked is not None:
                    locked.chmod(0o700)
        want_stop = label != "a sound tree with a symbolic link"
        out.append((label, stopped == want_stop, f"{'STOPPED' if stopped else 'enumerated'}: {detail}"))
    return out


def _state_controls(S, record):
    """Synthetic registries and plans: the expected state must hold for the right ones and fail for
    every wrong one, in both modes."""
    lo, hi = record["block"]

    def reg(accounted, exposed="()", other="X = 1"):
        return (f"ACCOUNTED_SEED_INTERVALS = (\n{accounted})\nEXPOSED_SEED_INTERVALS = {exposed}\n"
                f"RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n{other}\n")
    old = f"    ({lo - 9000}, {lo - 8000}),\n"
    pre = reg(old)
    rec = {**record, "registry_sha256": _sha(pre.encode())}
    plan = {"mode": "pilot", "pairs": [{"pair_id": p["pair_id"], "seeds": {"red": p["red"], "black": p["black"]}}
                                       for p in record["pairs"]]}
    good = reg(old + f"    ({lo}, {hi}),   # the tactical-candidate pilot\n")
    wrong_seed = json.loads(json.dumps(plan))
    wrong_seed["pairs"][3]["seeds"]["black"] += 1
    cases = [
        ("pre: the survey's registry, no plan", "pre", pre, None, None, True),
        ("pre: the registry changed", "pre", reg(old, other="X = 2"), None, None, False),
        ("pre: the block already registered", "pre", good, None, None, False),
        ("pre: a plan already exists", "pre", pre, None, plan, False),
        ("post: exactly the block appended last, the plan", "post", good, pre, plan, True),
        ("post: the block appended FIRST", "post", reg(f"    ({lo}, {hi}),\n" + old), pre, plan, False),
        ("post: two tuples appended", "post", reg(old + f"    ({lo}, {hi}),\n    ({hi + 50}, {hi + 60}),\n"), pre,
         plan, False),
        ("post: a different tuple", "post", reg(old + f"    ({lo}, {hi + 1}),\n"), pre, plan, False),
        ("post: EXPOSED changed too", "post", reg(old + f"    ({lo}, {hi}),\n", exposed=f"(({lo - 9000}, {lo - 8999}),)"),
         pre, plan, False),
        ("post: another statement changed", "post", reg(old + f"    ({lo}, {hi}),\n", other="X = 2"), pre, plan, False),
        ("post: no plan", "post", good, pre, None, False),
        ("post: a plan seed that is not the survey's", "post", good, pre, wrong_seed, False),
        ("post: a plan of 15 pairs", "post", good, pre, {**plan, "pairs": plan["pairs"][:15]}, False),
    ]
    out = []
    for label, mode, cur, pre_text, p, want_ok in cases:
        faults = state_faults(mode, rec, cur, pre_text, p, p is not None, S)
        out.append((label, (not faults) == want_ok, "; ".join(faults) or "the expected state"))
    regs = {"ACCOUNTED": [(lo - 9000, lo - 8000), (lo, hi)], "EXPOSED": [], "RETIRED": [], "TEST_ONLY": []}
    out.append(("post: only the block's own last ACCOUNTED interval leaves the prior terms",
                prior_registries("post", regs, (lo, hi))["ACCOUNTED"] == [(lo - 9000, lo - 8000)] and
                prior_registries("pre", regs, (lo, hi)) == regs, "pre keeps it; post drops exactly it"))
    return out


def main(argv):
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else None
    if mode not in OUT:
        raise ProofRefused("--mode must be pre or post")
    out = ROOT / OUT[mode]
    if out.exists():
        raise ProofRefused(f"{OUT[mode]} exists: each mode runs once (create-only)")
    S = load_survey()
    record = read_record((ROOT / SURVEY_RECORD).read_text(encoding="utf-8"), S)
    block = record["block"]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    say(f"TACTICAL CANDIDATE A8.8 STEP 5 -- COLLISION PROOF, mode {mode.upper()}")
    say("=" * 96)
    say(f"HEAD {head}; working tree {'DIRTY' if dirty else 'clean'}")
    say(f"survey record {SURVEY_RECORD} sha256 {SURVEY_RECORD_SHA256} (pinned; read at {record['head']})")
    say(f"survey script {SURVEY_SCRIPT} sha256 {SURVEY_SCRIPT_SHA256} (pinned; its scope, exemptions and check)")
    say(f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {record['base']}; {len(record['pairs'])} pairs")
    cur_text = (ROOT / S.REGISTRY_SOURCE).read_text(encoding="utf-8")
    pre_text = cur_text if mode == "pre" else subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{record['head']}:{S.REGISTRY_SOURCE}"],
        capture_output=True, text=True, check=True).stdout
    plan_path = ROOT / S.PLAN
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", S.PLAN],
                             capture_output=True).returncode == 0
    say(f"registry sha256 now {_sha(cur_text.encode())}; the survey read {record['registry_sha256']}")
    faults = state_faults(mode, record, cur_text, pre_text, plan, tracked, S)
    say(f"EXPECTED STATE ({mode}): {'HOLDS' if not faults else 'FAILS -- ' + '; '.join(faults)}")
    registries, named, gen, masks, consumed = S.prior_terms()
    registries = prior_registries(mode, registries, block)
    try:
        files, links = strict_files(S)
        literals = S.repo_literals(files, block)
    except S.SurveyStop as e:
        say(f"\nPROOF STOPPED (fail-closed): {e} -- NO VERDICT")
        _write(out)
        return 2
    say(f"\nSCOPE (fail-closed enumeration): {len(files)} regular files under {list(S.ROOTS)}; "
        f"symbolic links (not followed): {len(links)}; band literal values counted: {len(literals)}")
    prior = (registries, named, gen, consumed, literals)
    r = S.check(block, prior, masks)
    say(f"THE BLOCK {r['block']}  n={r['n']}  direct={r['direct'] or 'NONE'}  streams={r['streams']}  "
        f"injective={r['injective']}  gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    say("\nCONTROLS")
    say("-" * 96)
    negatives = [(f"{k} block {iv}", iv) for k, ivs in named.items() for iv in ivs[:1]]
    negatives += [("the TEST_ONLY band", registries["TEST_ONLY"][0]),
                  ("inside the live H3 uniform generation range", (gen[-1][0], gen[-1][0] + S.N)),
                  ("overlaps the block by one seed", (block[1] - 1, block[1] + S.N - 1)),
                  ("starts where the block ends -- inside the floor", (block[1], block[1] + S.N)),
                  ("one seed inside the floor", (block[1] + S.N - 1, block[1] + 2 * S.N - 1))]
    mutation = S.mutation_literal()
    negatives.append((f"a block at the H4-4b MUTATION literal {mutation}", (mutation, mutation + S.N)))
    results = [(f"REJECTED: {why}", not S.check(b, prior, masks, others=(block,))["clean"], "") for why, b in negatives]
    results += S._plant_controls(block, prior, masks, literals, r["clean"])
    results += S._exemption_controls(block)
    results += _enumeration_controls(S)
    results += _state_controls(S, record)
    passed = 0
    for label, ok, detail in results:
        passed += ok
        say(f"  {'PASS' if ok else 'FAIL':6s} {label}{'  [' + detail + ']' if detail else ''}")
    say(f"\ncontrols passed: {passed}/{len(results)}")
    ok, text = verdict(r["clean"], faults, passed, len(results), bool(dirty))
    say(f"PROOF VERDICT ({mode}): {text}")
    _write(out)
    return 0 if ok else 1


def _write(out):
    with open(out, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except ProofRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
