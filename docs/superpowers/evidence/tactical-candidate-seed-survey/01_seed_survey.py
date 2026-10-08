"""TACTICAL CANDIDATE, A8.8 STEP 4 -- THE FRESH, READ-ONLY PILOT SEED SURVEY (one 32-seed block).

Card `docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §18 A8.1-A8.2 (corrected
twice). NOT THE COLLISION PROOF: the proof re-runs against the registries as they stand then.
Nothing is drawn, reserved, registered, exposed or consumed; no model, JVM or game.

METHOD -- H5's survey (`evidence/2026-09-27-t1j-h5-seed-survey/01_h5_seed_survey.py`), i.e. H4's
collision proof v16 with the union scanner, applied FRESH: no survey output, candidate, exclusion
list or cleanliness claim of H4 or H5 is inherited.
  * every candidate seed and its four XOR-mask derivations disjoint from every prior term: the four
    registries, CONSUMED_SEEDS, the named spent blocks (D1, H1, H2, H3 with its retired segment
    blocks, H4's five, H5's four), H3's generation ranges, and every literal in the band;
  * the derivations injective across the block; the nearest other interval at least 32 away.

SCOPE (A8.1) -- H5's roots, NOT narrowed: every regular file under scripts/, tests/ and
docs/superpowers/, whatever its git state or suffix. `.gz` is decompressed first. Text is read as
UTF-8 with each undecodable byte REPLACED (a non-digit, so no digit run is joined or split). An
unreadable file or a failed decompression STOPS the survey with no verdict. Stated, not claimed:
an integer stored in binary form (.pyc, .class) cannot be seen by a decimal scan.

EXEMPTIONS (A8.2), exact paths fixed in the card before any run: this survey's two records, the
collision proof's three, the pilot plan only if every band literal in it is a seed of the block,
and -- by occurrence -- the (lo, hi) integer elements of the four registry assignments.

THE RULE (A8.1), fixed before any result: BASE = the first multiple of 10,000 strictly above the
highest seed of every registered or named-spent interval in the nine-digit 2026 band; the pilot
block = [BASE, BASE + 32); pair k = pilot-{k:02d}, red BASE + 2k, black BASE + 2k + 1. Checked,
never adjusted: NOT CLEAN -> stop. This script holds no seed literal; it does nothing on import.

    python <this file> --out <create-only run record>
"""
import ast
import bisect
import gzip
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
ROOTS = ("scripts", "tests", "docs/superpowers")
HERE = "docs/superpowers/evidence/tactical-candidate-seed-survey"
PROOF = "docs/superpowers/evidence/tactical-candidate-seed-registration"
PLAN = "docs/superpowers/evidence/tactical-candidate-plans/pilot-s0.json"
#: 🔴 A8.2 -- exact paths, not prefixes; nothing else is exempt
OWN_RECORDS = (f"{HERE}/01_seed_survey.py", f"{HERE}/02_seed_survey_run.txt",
               f"{PROOF}/01_collision_proof.py", f"{PROOF}/02_collision_proof_pre_run.txt",
               f"{PROOF}/03_collision_proof_post_run.txt")
REGISTRY_SOURCE = "scripts/GPU/alphazero/e4_screen_reference.py"
REGISTRY_NAMES = ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                  "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")
BAND = (2026 * 10 ** 5, 2027 * 10 ** 5)          # the nine-digit 2026 band
N, ALIGN, PAIRS = 32, 10_000, 16                 # A8.1: 16 pairs, one seed per colour

V16 = re.compile(r"(?<![\d.])(2026\d{5})(?!\d)")
GROUPED = re.compile(r"(?<![\d.])(\d(?:_?\d)*)(?![\d_])")


class SurveyStop(Exception):
    """A8.1: an unreadable file or a failed decompression -- the survey stops, no verdict."""


# ─────────────────────────────── the scope (A8.1) ───────────────────────────────

def repo_files(root=ROOT):
    """Every regular file under the three roots, whatever its suffix or git state. Symbolic
    links are not followed; they are returned separately (and reported)."""
    files, links = [], []
    for base in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root / base, followlinks=False):
            for name in sorted(dirnames):
                if os.path.islink(os.path.join(dirpath, name)):
                    links.append(pathlib.Path(dirpath) / name)
            for name in sorted(filenames):
                p = pathlib.Path(dirpath) / name
                (links if p.is_symlink() else files).append(p)
    return sorted(files), sorted(links)


def read_text(path, *, errors="replace"):
    """The file's text, `.gz` decompressed; fail-closed (A8.1)."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise SurveyStop(f"{path}: unreadable ({type(e).__name__}: {e})") from None
    if path.suffix == ".gz":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError, zlib.error) as e:
            raise SurveyStop(f"{path}: decompression failed ({type(e).__name__}: {e})") from None
    return raw.decode("utf-8", errors=errors)


def registry_tuple_occurrences(text):
    """(line, column-in-characters) of every int element of a (lo, hi) tuple in the four registry
    assignments -- by PARSING, never by value (v16's rule; ast's byte offsets converted)."""
    lines = text.splitlines(True)
    out = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in REGISTRY_NAMES for t in node.targets):
            for t in ast.walk(node.value):
                if isinstance(t, ast.Tuple) and len(t.elts) == 2:
                    for e in t.elts:
                        if isinstance(e, ast.Constant) and type(e.value) is int:
                            line = lines[e.lineno - 1].encode("utf-8")
                            out.add((e.lineno, len(line[:e.col_offset].decode("utf-8"))))
    return out


def scan(text, exempt=frozenset(), *, grouped=True):
    """{value: count} of band literals at every occurrence. `grouped=False` is v16's scanner
    alone -- kept only to REPORT what it would miss."""
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    hits = {}

    def keep(pos, value):
        line = bisect.bisect_right(starts, pos)
        if (line, pos - starts[line - 1]) not in exempt:
            hits[value] = hits.get(value, 0) + 1
    seen = set()
    for m in V16.finditer(text):
        seen.add(m.start())
        keep(m.start(), int(m.group(1)))
    if grouped:
        for m in GROUPED.finditer(text):
            digits = m.group(1).replace("_", "")
            if "_" in m.group(1) and len(digits) == 9 and digits.startswith("2026") and m.start() not in seen:
                keep(m.start(), int(digits))
    return hits


def repo_literals(files, block, root=ROOT, *, grouped=True, errors="replace"):
    """{value: [(rel, count)]} over `files`, the A8.2 exemptions applied; raises SurveyStop."""
    out = {}
    for p in files:
        try:
            rel = str(p.resolve().relative_to(root.resolve()))
        except ValueError:
            rel = str(p)                          # outside the root: never exempt
        if rel in OWN_RECORDS:
            continue
        text = read_text(p, errors=errors)
        exempt = registry_tuple_occurrences(text) if rel == REGISTRY_SOURCE else frozenset()
        hits = scan(text, exempt, grouped=grouped)
        if rel == PLAN and hits and set(hits) <= set(range(*block)):
            continue                              # A8.2: the plan, exempt only if it is the block
        for v, n in hits.items():
            out.setdefault(v, []).append((rel, n))
    return out


# ─────────────────────────────── the prior terms and the rule ───────────────────────────────

def prior_terms():
    """The registries, CONSUMED_SEEDS, the named spent blocks, H3's generation ranges and the
    masks -- all READ, as they stand now."""
    sys.path.insert(0, str(ROOT))
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import h3_study_rules as H3R
    from scripts.GPU.alphazero import h4_runner as R4
    from scripts.GPU.alphazero import h5_runner as H5
    from scripts.GPU.alphazero.d1_selection import SEED_INTERVAL as D1_PAPER
    from scripts.GPU.alphazero.h1_viability_rules import H1_ATTEMPT1_SEED_BLOCK, H1_SEED_BLOCK
    from scripts.GPU.alphazero.h2_match_rules import H2_ATTEMPT1_SEED_BLOCK, H2_ATTEMPT2_SEED_BLOCK, H2_SEED_BLOCK
    from scripts.GPU.alphazero.h3_pilot_runner import PILOT_SEED_BLOCK as H3_PILOT
    from scripts.GPU.alphazero.h3_study_runner import RETIRED_SEGMENT_BLOCKS, SEGMENT_SEED_BLOCKS
    from scripts.GPU.alphazero.twixtbot_g3_reference import SeededReferenceAgent
    from scripts.GPU.alphazero.twixtbot_g3_schedule import CONSUMED_SEEDS
    registries = {"ACCOUNTED": [tuple(iv) for iv in REG.ACCOUNTED_SEED_INTERVALS],
                  "EXPOSED": [tuple(iv) for iv in REG.EXPOSED_SEED_INTERVALS],
                  "RETIRED": [tuple(iv) for iv in REG.RETIRED_SEED_INTERVALS],
                  "TEST_ONLY": [tuple(iv) for iv in REG.TEST_ONLY_SEED_INTERVALS]}
    named = {"D1 paper": [tuple(D1_PAPER)], "H1": [tuple(H1_ATTEMPT1_SEED_BLOCK), tuple(H1_SEED_BLOCK)],
             "H2": [tuple(b) for b in (H2_ATTEMPT1_SEED_BLOCK, H2_ATTEMPT2_SEED_BLOCK, H2_SEED_BLOCK)],
             "H3": [tuple(H3_PILOT), *(tuple(b) for b in SEGMENT_SEED_BLOCKS),
                    *(tuple(b) for b in RETIRED_SEGMENT_BLOCKS)],
             "H4": [tuple(b) for b in R4.SEED_BLOCKS.values()],
             "H5": [tuple(b) for b in H5.SEED_BLOCKS.values()]}
    gen = [(lo, hi) for lo, hi, _why in H3R.RETIRED_GENERATION_RANGES]
    gen.append(tuple(H3R.generation_seed_range(H3R.GEN_SEED_UNIFORM)))
    masks = sorted({*SeededReferenceAgent.SEARCH_MASK.values(), *SeededReferenceAgent.READOUT_MASK.values()})
    assert len(masks) == 4, masks
    return registries, named, gen, masks, set(CONSUMED_SEEDS)


def candidate(intervals):
    """A8.1, exactly: the first multiple of 10,000 STRICTLY above the highest SEED (an interval's
    last seed is hi - 1) of every registered or named-spent interval lying in the band."""
    highest = max(hi - 1 for lo, hi in intervals if BAND[0] <= lo < BAND[1])
    base = (highest // ALIGN + 1) * ALIGN
    block = (base, base + N)
    assert block[1] <= BAND[1], "the rule left the 2026 band"
    return highest, base, block


def pair_seeds(block):
    """A8.1: pilot-k plays red BASE + 2k and black BASE + 2k + 1 in BOTH games of its pair."""
    lo = block[0]
    return [{"pair_id": f"pilot-{k:02d}", "red": lo + 2 * k, "black": lo + 2 * k + 1} for k in range(PAIRS)]


def intervals_set(ivs):
    return {s for lo, hi in ivs for s in range(lo, hi)}


def derivations(seeds, masks):
    return {v for s in seeds for v in (s, *(s ^ m for m in masks))}


def check(block, prior, masks, others=()):
    """v16's check: direct overlap, derived-stream overlap, injectivity, the gap floor (32)."""
    registries, named, gen, consumed, literals = prior
    lits = set(literals)
    ours = set(range(*block))
    cats = {k: intervals_set(v) for k, v in registries.items()}
    cats.update({f"NAMED {k}": intervals_set(v) for k, v in named.items()})
    cats.update({"CONSUMED_SEEDS": set(consumed), "REPO_LITERALS": lits, "OTHER": intervals_set(others)})
    for i, g in enumerate(gen):
        cats[f"GEN_RANGE_{i}"] = set(range(*g))
    union = set().union(*cats.values())
    vals = derivations(ours, masks)
    floor = block[1] - block[0]
    ivs = ([iv for v in registries.values() for iv in v] + [iv for v in named.values() for iv in v]
           + list(gen) + list(others) + [(s, s + 1) for s in set(consumed) | lits])
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi for lo, hi in ivs if not (lo < block[1] and block[0] < hi)]
    direct = {k: len(ours & v) for k, v in cats.items() if ours & v}
    streams = len(vals & derivations(union, masks))
    nearest = min(gaps)
    return {"block": block, "n": len(ours), "direct": direct, "streams": streams,
            "injective": len(vals) == len(ours) * 5, "nearest_gap": nearest, "floor": floor,
            "clean": not direct and not streams and len(vals) == len(ours) * 5 and nearest >= floor}


def git_states(root=ROOT):
    """rel -> tracked | untracked | ignored, for every file under the three roots."""
    def ls(*args):
        r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", *args, "--", *ROOTS],
                           capture_output=True, check=True)
        return {x for x in r.stdout.decode().split("\0") if x}
    return ls(), ls("--others", "--exclude-standard")


def mutation_literal(root=ROOT):
    """The planted wrong block of the H4 4b control 'a seed block is not the registered one' --
    READ from the control list, not typed here (as H5)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_controls", root / "scripts/GPU/alphazero/injected_defect_controls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (new,) = [d[3] for d in mod.DEFECTS if d[0] == "H4 4b: a seed block is not the registered one"]
    return [int(m.group(1).replace("_", "")) for m in GROUPED.finditer(new) if len(m.group(1).replace("_", "")) == 9][0]


# ─────────────────────────────── the run ───────────────────────────────

_LINES = []


def say(*a):
    print(*a, flush=True)
    _LINES.append(" ".join(str(x) for x in a))


def _plant_controls(block, prior, masks, base_literals, base_clean):
    """Differential controls, each in a temporary tree OUTSIDE the repository: without the plant
    the block is CLEAN; with it, it fails ON the plant. Then the fail-closed controls."""
    lo, hi = block
    results = []
    plants = [("an UNDERSCORE-GROUPED candidate seed (.py)", "notes.py", f"STRAY = {lo:_}\n".encode(), "direct"),
              ("a PLAIN candidate seed (.py)", "notes.py", f"STRAY = {lo + 7}\n".encode(), "direct"),
              ("an UNDERSCORE-GROUPED exclusive end (.py)", "notes.py", f"STRAY = {hi:_}\n".encode(), "gap"),
              ("a candidate seed in a .log -- a suffix H5's filter skipped", "run.log",
               f"seed={lo + 5}\n".encode(), "direct"),
              ("a candidate seed inside a .gz", "trace.jsonl.gz", gzip.compress(f'{{"seed": {lo + 9}}}\n'.encode()),
               "direct"),
              ("a candidate seed after an undecodable byte after a digit", "blob.bin",
               b"7\xff" + str(lo + 3).encode() + b"\n", "direct")]
    for label, name, raw, kind in plants:
        with tempfile.TemporaryDirectory() as tmp:
            f = pathlib.Path(tmp) / name
            f.write_bytes(raw)
            mine = repo_literals([f], block, pathlib.Path(tmp))
            r = check(block, (*prior[:4], {**base_literals, **mine}), masks)
            ignore_view = repo_literals([f], block, pathlib.Path(tmp), errors="ignore")
        failed_on = r["direct"].get("REPO_LITERALS", 0) >= 1 if kind == "direct" else \
            (not r["direct"] and r["nearest_gap"] == 0 < r["floor"])
        results.append((label, base_clean and failed_on and not r["clean"],
                        f"without it CLEAN={base_clean}; with it direct={r['direct'] or 'NONE'} "
                        f"gap={r['nearest_gap']}; dropping undecodable bytes "
                        f"instead would {'see' if ignore_view else 'MISS'} it"))
    for label, name, raw in (("an UNREADABLE file stops the survey", "locked.txt", b"x"),
                             ("a CORRUPT .gz stops the survey", "broken.gz", b"not gzip at all")):
        with tempfile.TemporaryDirectory() as tmp:
            f = pathlib.Path(tmp) / name
            f.write_bytes(raw)
            if name == "locked.txt":
                f.chmod(0)
            try:
                repo_literals([f], block, pathlib.Path(tmp))
                stopped = False
            except SurveyStop:
                stopped = True
            finally:
                f.chmod(0o600)
        results.append((label, stopped, "SurveyStop" if stopped else "NOT STOPPED"))
    return results


def _exemption_controls(block):
    """A8.2 on a synthetic tree: an own record and the plan of the block are exempt; the same
    literal elsewhere, or a plan with a seed outside the block, is not; registry tuples by parsing."""
    lo = block[0]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        files = {OWN_RECORDS[3]: f"pre-proof saw {lo + 1}\n",
                 f"{PROOF}/04_not_listed.txt": f"saw {lo + 2}\n",
                 PLAN: f'{{"red": {lo}, "black": {lo + 1}}}\n',
                 REGISTRY_SOURCE: f'ACCOUNTED_SEED_INTERVALS = (("é" * 0) or ({lo + 10}, {lo + 12}),)\n'
                                  f"NOTE = {lo + 11}\n"}
        for rel, text in files.items():
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            (t / rel).write_text(text)
        lits = repo_literals([t / rel for rel in files], block, t)
        results.append(("an own record (exact path) is exempt", lo + 1 not in lits, str(lits.get(lo + 1))))
        results.append(("a sibling path that is NOT listed is counted", lo + 2 in lits, str(lits.get(lo + 2))))
        results.append(("the plan holding only block seeds is exempt", lo not in lits, str(lits.get(lo))))
        results.append(("registry tuple elements are exempt by parsing (a non-ASCII char BEFORE them)",
                        lo + 10 not in lits and lo + 12 not in lits, str(sorted(v for v in lits if v >= lo + 10))))
        results.append(("the same value outside a registry tuple is counted", lo + 11 in lits, str(lits.get(lo + 11))))
        (t / PLAN).write_text(f'{{"red": {lo}, "black": {lo + N + 5}}}\n')
        lits = repo_literals([t / PLAN], block, t)
        results.append(("a plan with a seed OUTSIDE the block is not exempt", lo in lits and lo + N + 5 in lits,
                        str(sorted(lits))))
    return results


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    if out is None or pathlib.Path(out).exists():
        raise SystemExit("--out names a create-only run record that must not exist")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    say("TACTICAL CANDIDATE A8.8 STEP 4 -- FRESH READ-ONLY PILOT SEED SURVEY (not the collision proof)")
    say("=" * 96)
    say(f"HEAD {head}; working tree {'DIRTY' if dirty else 'clean'}")
    say(f"registry {REGISTRY_SOURCE} sha256 {hashlib.sha256((ROOT / REGISTRY_SOURCE).read_bytes()).hexdigest()}")
    registries, named, gen, masks, consumed = prior_terms()
    say(f"masks (imported): {[hex(m) for m in masks]}")
    say("registries: " + ", ".join(f"{k} {len(v)}" for k, v in registries.items()))
    say("named spent blocks: " + ", ".join(f"{k} {len(v)}" for k, v in named.items()) +
        f"; H3 generation ranges {len(gen)}; CONSUMED_SEEDS {len(consumed)}")
    say(f"\nEXEMPT PATHS (A8.2; exact; plus registry tuple elements by parsing): {len(OWN_RECORDS)} + the plan")
    for p in (*OWN_RECORDS, f"{PLAN}  (only if every band literal in it is a seed of the block)"):
        say(f"    {p}")
    reserved = [iv for v in registries.values() for iv in v] + [iv for v in named.values() for iv in v]
    highest, base, block = candidate(reserved)
    say(f"\nCANDIDATE RULE: highest registered/named-spent SEED in the band = {highest}; BASE = {base}; "
        f"block = [{block[0]}, {block[1]}) -- {N} seeds")
    files, links = repo_files()
    tracked, untracked = git_states()
    say(f"\nSCOPE: {len(files)} regular files under {list(ROOTS)}; symbolic links (not followed): {len(links)}")
    by = {}
    for p in files:
        rel = str(p.relative_to(ROOT))
        state = "tracked" if rel in tracked else "untracked" if rel in untracked else "ignored"
        key = (p.suffix or "<none>", state)
        by[key] = by.get(key, 0) + 1
    for (suffix, state), n in sorted(by.items(), key=lambda kv: (-kv[1], kv[0])):
        say(f"    {suffix:14s} {state:9s} {n}")
    try:
        literals = repo_literals(files, block)
    except SurveyStop as e:
        say(f"\nSURVEY STOPPED (A8.1): {e} -- NO VERDICT")
        _write(out)
        return 2
    v16_only = repo_literals(files, block, grouped=False)
    say(f"\nband literal values counted (union scanner, replaced bytes): {len(literals)}; v16's scanner alone: "
        f"{len(v16_only)}; values only the underscore-aware scan finds: {len(set(literals) - set(v16_only))}")
    in_band = sorted(v for v in literals if BAND[0] <= v < BAND[1])
    say(f"band literal values: {len(in_band)}; the highest: {in_band[-1] if in_band else None}")
    prior = (registries, named, gen, consumed, literals)
    r = check(block, prior, masks)
    say(f"\nTHE BLOCK {r['block']}  n={r['n']}  direct={r['direct'] or 'NONE'}  streams={r['streams']}  "
        f"injective={r['injective']}  gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    say("PAIR SEEDS (A8.1; the seed follows the colour in both games of a pair):")
    for p in pair_seeds(block):
        say(f"    {p['pair_id']}  red {p['red']}  black {p['black']}")
    say("\nNEGATIVE CONTROLS -- each must be REJECTED")
    say("-" * 96)
    controls = [(f"{k} block {iv}", iv) for k, ivs in named.items() for iv in ivs[:1]]
    controls += [("the TEST_ONLY band", registries["TEST_ONLY"][0]),
                 ("inside the live H3 uniform generation range", (gen[-1][0], gen[-1][0] + N)),
                 ("overlaps the block by one seed", (block[1] - 1, block[1] + N - 1)),
                 ("starts where the block ends -- inside the floor", (block[1], block[1] + N)),
                 ("one seed inside the floor", (block[1] + N - 1, block[1] + 2 * N - 1))]
    mutation = mutation_literal()
    controls.append((f"a block at the H4-4b MUTATION literal {mutation} (injected_defect_controls.py)",
                     (mutation, mutation + N)))
    rejected, total = 0, 0
    for why, b in controls:
        rr = check(b, prior, masks, others=(block,))
        ok = not rr["clean"]
        rejected, total = rejected + ok, total + 1
        say(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {why}")
    say("\nDIFFERENTIAL AND FAIL-CLOSED CONTROLS (temporary trees outside the repository)")
    say("-" * 96)
    for label, ok, detail in _plant_controls(block, prior, masks, literals, r["clean"]):
        rejected, total = rejected + ok, total + 1
        say(f"  {'PASS' if ok else 'FAIL':12s} {label}: {detail}")
    edge_hi = base + ALIGN                       # an interval whose exclusive end is a multiple of 10,000
    _h, edge_base, _b = candidate([(edge_hi - 5, edge_hi)])
    ok = edge_base == edge_hi
    rejected, total = rejected + ok, total + 1
    say(f"  {'PASS' if ok else 'FAIL':12s} BASE when the last interval ENDS at a multiple of 10,000: the highest seed "
        f"is end - 1, so BASE = end ({edge_base == edge_hi})")
    say("\nEXEMPTION CONTROLS (A8.2, synthetic tree)")
    say("-" * 96)
    for label, ok, detail in _exemption_controls(block):
        rejected, total = rejected + ok, total + 1
        say(f"  {'PASS' if ok else 'FAIL':12s} {label}  [{detail}]")
    say(f"\ncontrols passed: {rejected}/{total}")
    ok = r["clean"] and rejected == total and not dirty
    say(f"SURVEY VERDICT   : {'THE PILOT BLOCK IS CLEAN' if r['clean'] else 'NOT CLEAN -- STOP (no adjustment)'}"
        f"{'' if rejected == total else ' -- A CONTROL FAILED'}{' -- RUN ON A DIRTY TREE' if dirty else ''}")
    _write(out)
    return 0 if ok else 1


def _write(out):
    with open(out, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
