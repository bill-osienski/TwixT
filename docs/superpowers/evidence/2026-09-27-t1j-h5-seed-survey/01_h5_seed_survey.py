"""H5 STEP 5 -- FRESH, READ-ONLY SEED SURVEY of four 148-seed candidate blocks.

H5 replication card §9 and implementation card §7. NOT THE COLLISION PROOF: step 6
re-runs a proof against the registries as they stand then. Nothing is drawn,
reserved, registered, exposed or consumed; no model, JVM or game. Imports, XOR
and set arithmetic over the registries AS THEY STAND NOW.

METHOD -- H4's collision proof v16 (provenance, `01c`), applied fresh:
  * every candidate seed and its four XOR-mask derivations disjoint from every
    prior term: all four registries, CONSUMED_SEEDS, the named spent blocks
    (D1, H1, H2, H3, and H4's five from `h4_runner.SEED_BLOCKS`), the H3
    generation ranges, the OTHER candidates, and every repository literal;
  * derivations injective; the nearest other interval at least the candidate's
    OWN size (148) away;
  * a literal counts at EVERY occurrence except (a) as an element of a (lo, hi)
    tuple inside one of the four registry assignments of e4_screen_reference.py,
    located by parsing, or (b) inside one of THIS survey's own pinned records.
  NOT INHERITED from H4: its candidate ranges, its survey output, its pinned
  exclusion list.

🔴 THE ONE CORRECTION (implementation card §7). v16 matched
`(?<![\\d.])(2026\\d{5})(?!\\d)`, which cannot see an underscore-grouped literal.
This scanner counts the UNION of that match and every maximal digit run WITH
OPTIONAL SINGLE UNDERSCORES between digits, normalized, that is nine digits
starting 2026 -- so it finds everything v16 finds and the underscored forms too.
A differential control plants an underscore-grouped candidate seed in an
unrelated file: the block must be CLEAN without it and fail ON IT with it.

THE CANDIDATE RULE -- fixed here, BEFORE any result exists:
  BASE = the first multiple of 10,000 strictly above the highest seed of every
         REGISTERED or NAMED-SPENT interval lying in the nine-digit 2026 band;
  block k (k = 0..3) = [BASE + 2,000 k, BASE + 2,000 k + 148).
The 2,000 spacing leaves each block more than its own floor from the next.
Candidates are CHECKED, never adjusted: if any is NOT CLEAN the survey says so
and step 5 stops. This script holds no seed literal of its own.

    python <this file> --out <create-only run record>
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys
import tempfile

from scripts.GPU.alphazero.e4_screen_reference import (
    ACCOUNTED_SEED_INTERVALS, EXPOSED_SEED_INTERVALS,
    TEST_ONLY_SEED_INTERVALS, RETIRED_SEED_INTERVALS)
from scripts.GPU.alphazero.twixtbot_g3_schedule import CONSUMED_SEEDS
from scripts.GPU.alphazero.twixtbot_g3_reference import SeededReferenceAgent
from scripts.GPU.alphazero.d1_selection import SEED_INTERVAL as D1_PAPER
from scripts.GPU.alphazero.h1_viability_rules import H1_ATTEMPT1_SEED_BLOCK as H1_A1
from scripts.GPU.alphazero.h1_viability_rules import H1_SEED_BLOCK as H1_A2
from scripts.GPU.alphazero.h2_match_rules import H2_SEED_BLOCK as H2_A3
from scripts.GPU.alphazero.h2_match_rules import H2_ATTEMPT1_SEED_BLOCK as H2_A1
from scripts.GPU.alphazero.h2_match_rules import H2_ATTEMPT2_SEED_BLOCK as H2_A2
from scripts.GPU.alphazero.h3_pilot_runner import PILOT_SEED_BLOCK as H3_PILOT
from scripts.GPU.alphazero.h3_study_runner import SEGMENT_SEED_BLOCKS, RETIRED_SEGMENT_BLOCKS
from scripts.GPU.alphazero import h3_study_rules as H3R
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_runner as H5

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None

HERE = "docs/superpowers/evidence/2026-09-27-t1j-h5-seed-survey"
#: 🔴 PINNED OWN RECORDS -- exact paths, not prefixes. This survey's two files, and
#: the step-6 proof's files under a FIXED (undated) directory named now, so the
#: exemption is fixed before any proof result exists. Nothing else is exempt.
OWN_RECORDS = (
    f"{HERE}/01_h5_seed_survey.py",
    f"{HERE}/02_h5_seed_survey_run.txt",
    "docs/superpowers/evidence/t1j-h5-seed-registration/01_h5_collision_proof.py",
    "docs/superpowers/evidence/t1j-h5-seed-registration/02_h5_collision_proof_pre_run.txt",
    "docs/superpowers/evidence/t1j-h5-seed-registration/03_h5_collision_proof_post_run.txt",
)
REGISTRY_SOURCE = "scripts/GPU/alphazero/e4_screen_reference.py"
REGISTRY_NAMES = ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                  "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")
BAND = (2026 * 10 ** 5, 2027 * 10 ** 5)          # the nine-digit 2026 band
N, SPACING, ALIGN = 2 * H5.PAIRS_PER_SEGMENT, 2_000, 10_000
assert N == 148

V16 = re.compile(r"(?<![\d.])(2026\d{5})(?!\d)")
GROUPED = re.compile(r"(?<![\d.])(\d(?:_?\d)*)(?![\d_])")


def registry_tuple_occurrences(text):
    """(line, col) of every int element of a (lo, hi) tuple in the four registry
    assignments -- by PARSING, never by value (v16's rule, unchanged)."""
    out = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) in REGISTRY_NAMES for t in node.targets):
            for t in ast.walk(node.value):
                if isinstance(t, ast.Tuple) and len(t.elts) == 2:
                    for e in t.elts:
                        if isinstance(e, ast.Constant) and type(e.value) is int:
                            out.add((e.lineno, e.col_offset))
    return out


def repo_files():
    return [p for base in ("scripts", "tests", "docs/superpowers")
            for p in (ROOT / base).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md", ".json", ".jsonl", ".txt")]


def scan(text, exempt=frozenset(), *, grouped=True):
    """{value: count} of literal occurrences, by provenance. `grouped=False` is
    v16's scanner alone -- kept only to REPORT what it would have missed."""
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    import bisect
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
            if "_" in m.group(1) and len(digits) == 9 and digits.startswith("2026") \
                    and m.start() not in seen:
                keep(m.start(), int(digits))
    return hits


def repo_literals(files=None, *, grouped=True):
    out = {}
    for p in (repo_files() if files is None else files):
        try:
            rel = str(p.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(p)                          # outside the repository: never exempt
        if rel in OWN_RECORDS:
            continue
        text = p.read_text(errors="ignore")
        exempt = registry_tuple_occurrences(text) if rel == REGISTRY_SOURCE else frozenset()
        for v, n in scan(text, exempt, grouped=grouped).items():
            out.setdefault(v, []).append((rel, n))
    return out


REGISTRIES = {"ACCOUNTED": [tuple(iv) for iv in ACCOUNTED_SEED_INTERVALS],
              "EXPOSED": [tuple(iv) for iv in EXPOSED_SEED_INTERVALS],
              "RETIRED": [tuple(iv) for iv in RETIRED_SEED_INTERVALS],
              "TEST_ONLY": [tuple(iv) for iv in TEST_ONLY_SEED_INTERVALS]}
NAMED_SPENT = (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3,
               *SEGMENT_SEED_BLOCKS, *RETIRED_SEGMENT_BLOCKS, *R4.SEED_BLOCKS.values())
MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS
GEN_RANGES = [(lo, hi) for lo, hi, _why in H3R.RETIRED_GENERATION_RANGES]
GEN_RANGES.append(H3R.generation_seed_range(H3R.GEN_SEED_UNIFORM))
LITERALS = repo_literals()


def intervals_set(ivs):
    return {s for lo, hi in ivs for s in range(lo, hi)}


def derivations(seeds):
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


def check(block, others=(), literals=None):
    """v16's check: direct overlap, derived-stream overlap, injectivity, gap floor."""
    lits = set(LITERALS if literals is None else literals)
    ours = set(range(*block))
    cats = {k: intervals_set(v) for k, v in REGISTRIES.items()}
    cats.update({"CONSUMED_SEEDS": set(CONSUMED_SEEDS), "NAMED_SPENT": intervals_set(NAMED_SPENT),
                 "REPO_LITERALS": lits, "OTHER_CANDIDATES": intervals_set(others)})
    for i, gr in enumerate(GEN_RANGES):
        cats[f"GEN_RANGE_{i}"] = set(range(*gr))
    prior = set().union(*cats.values())
    vals = derivations(ours)
    floor = block[1] - block[0]
    ivs = ([iv for v in REGISTRIES.values() for iv in v] + list(NAMED_SPENT)
           + list(GEN_RANGES) + list(others) + [(s, s + 1) for s in set(CONSUMED_SEEDS) | lits])
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ivs if not (lo < block[1] and block[0] < hi)]
    direct = {k: len(ours & v) for k, v in cats.items() if ours & v}
    streams = len(vals & derivations(prior))
    nearest = min(gaps)
    return {"block": block, "n": len(ours), "direct": direct, "streams": streams,
            "injective": len(vals) == len(ours) * 5, "nearest_gap": nearest, "floor": floor,
            "clean": not direct and not streams and len(vals) == len(ours) * 5
            and nearest >= floor}


def candidates():
    """THE RULE of the docstring, and nothing else."""
    reserved = [iv for v in REGISTRIES.values() for iv in v] + list(NAMED_SPENT)
    top = max(hi for lo, hi in reserved if BAND[0] <= lo < BAND[1])
    base = (top // ALIGN + 1) * ALIGN
    blocks = {f"H5_STUDY_SEGMENT_{k}": (base + SPACING * k, base + SPACING * k + N)
              for k in range(4)}
    assert max(hi for _lo, hi in blocks.values()) <= BAND[1], "the rule left the 2026 band"
    return top, base, blocks


def mutation_literal():
    """The planted wrong block of the H4 4b control 'a seed block is not the
    registered one' -- READ from the control list, not typed here."""
    import importlib.util
    path = ROOT / "scripts/GPU/alphazero/injected_defect_controls.py"
    spec = importlib.util.spec_from_file_location("_controls", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (new,) = [d[3] for d in mod.DEFECTS if d[0] == "H4 4b: a seed block is not the registered one"]
    (value,) = [int(m.group(1).replace("_", "")) for m in GROUPED.finditer(new)
                if len(m.group(1).replace("_", "")) == 9][:1]
    return value


_LINES = []


def say(*a):
    print(*a)
    _LINES.append(" ".join(str(x) for x in a))


def _sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


if __name__ == "__main__":
    if OUT is None or pathlib.Path(OUT).exists():
        raise SystemExit("--out names a create-only run record that must not exist")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                            "scripts", "tests"], capture_output=True, text=True).stdout.strip()
    say("H5 STEP 5 -- FRESH READ-ONLY SEED SURVEY (not the collision proof)")
    say("=" * 72)
    say(f"HEAD {head}; scripts/ and tests/ {'DIRTY' if dirty else 'clean'}")
    say(f"registry {REGISTRY_SOURCE} sha256 {_sha(REGISTRY_SOURCE)}")
    say(f"masks (imported): {[hex(m) for m in MASKS]}")
    say("registries: " + ", ".join(f"{k} {len(v)}" for k, v in REGISTRIES.items()))
    say(f"named spent blocks: {len(NAMED_SPENT)} (H4's five included); "
        f"H3 generation ranges: {len(GEN_RANGES)}")
    say("")
    say(f"PINNED OWN RECORDS (exact paths; the ONLY files whose literals are exempt): "
        f"{len(OWN_RECORDS)}")
    for p in OWN_RECORDS:
        say(f"    {p}")
    say("")
    v16_only = repo_literals(grouped=False)
    grouped_only = sorted(set(LITERALS) - set(v16_only))
    say(f"repo literal values counted (union scanner): {len(LITERALS)}; "
        f"v16's scanner alone: {len(v16_only)}")
    say(f"values ONLY the underscore-aware scan finds: {len(grouped_only)}")
    for v in grouped_only:
        say(f"    {v}  in {[f for f, _n in LITERALS[v]]}")
    in_band = sorted(v for v in LITERALS if BAND[0] <= v < BAND[1])
    say(f"literal values in the 2026 band: {len(in_band)}")
    say("")
    top, base, blocks = candidates()
    say(f"CANDIDATE RULE: highest registered/named seed end in the band = {top}; "
        f"BASE = {base}; spacing {SPACING}; size {N}")
    results = {}
    for name, block in blocks.items():
        others = tuple(b for n, b in blocks.items() if n != name)
        r = results[name] = check(block, others)
        say(f"{name:20s} {r['block']}  n={r['n']}  direct={r['direct'] or 'NONE'}  "
            f"streams={r['streams']}  injective={r['injective']}  "
            f"gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    say(f"total seeds: {sum(r['n'] for r in results.values())} = 4 x {N}")
    say("")
    say("NEGATIVE CONTROLS -- each must be REJECTED")
    say("-" * 72)
    everything = tuple(blocks.values())
    seg0 = blocks["H5_STUDY_SEGMENT_0"]
    mutation = mutation_literal()
    controls = [
        ("H4 study segment 0 (exposed, retired)", R4.SEED_BLOCKS[("study", 0)]),
        ("H4 pilot block", R4.SEED_BLOCKS[("pilot", 0)]),
        ("H3 segment-0 replacement block (spent)", SEGMENT_SEED_BLOCKS[0]),
        ("H3's retired segment-0 quarter", RETIRED_SEGMENT_BLOCKS[0]),
        ("the H3 pilot block", H3_PILOT),
        ("H2 attempt-3 block", H2_A3),
        ("D1's paper reservation", D1_PAPER),
        ("the TEST_ONLY band", REGISTRIES["TEST_ONLY"][0]),
        ("inside the live H3 uniform generation range",
         (GEN_RANGES[-1][0], GEN_RANGES[-1][0] + N)),
        ("overlaps candidate 0 by one seed", (seg0[1] - 1, seg0[1] + N - 1)),
        ("starts where candidate 0 ends -- inside the floor", (seg0[1], seg0[1] + N)),
        ("one seed inside candidate 0's floor", (seg0[1] + N - 1, seg0[1] + 2 * N - 1)),
    ]
    controls.append((f"a block at the H4-4b MUTATION literal {mutation} "
                     f"(injected_defect_controls.py)", (mutation, mutation + N)))
    rejected = 0
    for why, block in controls:
        r = check(block, everything)
        ok = not r["clean"]
        rejected += ok
        say(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {why}  {block}")
    v16_view = check((mutation, mutation + N), everything, literals=v16_only)
    say(f"      descriptive: with v16's scanner alone that block is "
        f"{'REJECTED' if not v16_view['clean'] else 'CLEAN'} "
        f"(direct={v16_view['direct'] or 'NONE'}) -- v16 now sees the value only "
        f"where it is written plainly: {[f for f, _n in v16_only.get(mutation, [])]}")
    # 🔴 DIFFERENTIAL controls: the same candidate, with and without a literal
    # planted in an UNRELATED file outside the repository. Without it: CLEAN.
    # With it: failed ON the plant. The first is the underscore-grouped form v16
    # could not see; v16's own scanner is shown NOT to see it.
    lo, hi = seg0
    others0 = tuple(b for n, b in blocks.items() if n != "H5_STUDY_SEGMENT_0")
    base_r = check(seg0, others0)
    plants = [("an UNDERSCORE-GROUPED candidate seed", f"{lo:_}",
               lambda r: r["direct"].get("REPO_LITERALS", 0) >= 1),
              ("a PLAIN candidate seed", str(lo + 7),
               lambda r: r["direct"].get("REPO_LITERALS", 0) >= 1),
              ("an UNDERSCORE-GROUPED exclusive end", f"{hi:_}",
               lambda r: not r["direct"] and r["nearest_gap"] == 0 < r["floor"])]
    for label, text, failed_on in plants:
        with tempfile.TemporaryDirectory() as tmp:
            unrelated = pathlib.Path(tmp) / "unrelated_notes.py"
            unrelated.write_text(f"STRAY = {text}\n")
            with_plant = {**LITERALS, **repo_literals([unrelated])}
            r = check(seg0, others0, literals=with_plant)
            v16_sees = bool(scan(unrelated.read_text(), grouped=False))
        ok = base_r["clean"] and not r["clean"] and failed_on(r)
        rejected += ok
        controls.append((label, seg0))
        say(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {label} ({text}): without it "
            f"CLEAN={base_r['clean']}; with it direct={r['direct'] or 'NONE'} "
            f"gap={r['nearest_gap']}; v16's scanner alone sees it: {v16_sees}")
    say(f"\ncontrols rejected: {rejected}/{len(controls)}")
    clean = all(r["clean"] for r in results.values())
    ok = clean and rejected == len(controls) and not dirty
    say(f"SURVEY VERDICT   : {'ALL FOUR CANDIDATES CLEAN' if clean else 'NOT CLEAN -- STOP'}"
        f"{'' if rejected == len(controls) else ' -- A CONTROL WAS NOT REJECTED'}"
        f"{' -- RUN ON A DIRTY TREE' if dirty else ''}")
    with open(OUT, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(_LINES) + "\n")
    sys.exit(0 if ok else 1)
