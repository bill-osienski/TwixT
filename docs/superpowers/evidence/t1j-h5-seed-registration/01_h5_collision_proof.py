"""H5 STEP 6 -- COLLISION PROOF for the four H5 study blocks. FROZEN; NOT YET RUN.

Implementation card §7 and §11 A2. The SAME script runs twice, with the SAME rule:

    PYTHONPATH=. .venv/bin/python <this file> pre  --out <dir>/02_h5_collision_proof_pre_run.txt
    PYTHONPATH=. .venv/bin/python <this file> post --out <dir>/03_h5_collision_proof_post_run.txt

`pre`  -- before registration: every block absent from every registry, and
          `h5_runner.SEED_BLOCKS` empty. Must be CLEAN, or nothing is registered.
`post` -- after the four blocks enter `ACCOUNTED_SEED_INTERVALS` and
          `h5_runner.SEED_BLOCKS`: each registered as EXACTLY its own interval in
          ACCOUNTED only, SEED_BLOCKS equal to the blocks, and each block excluded
          from the priors BY IDENTITY for its own check (never by subtracting its
          seeds). Must be CLEAN.

THE BLOCKS are READ from the committed survey record, hash-pinned below -- this
file holds no seed literal. METHOD: the step-5 survey's, unchanged (H4 v16
provenance + the underscore-aware union scanner). A literal counts at EVERY
occurrence except:
  (a) an int element of a (lo, hi) tuple inside one of the four registry
      assignments of e4_screen_reference.py, located by parsing (v16, unchanged);
  (b) 🔴 AMENDMENT A2: an int element of the (lo, hi) tuple VALUE of one of the
      four segment entries (int keys 0, 1, 2, 3) of the `SEED_BLOCKS` assignment
      of scripts/GPU/alphazero/h5_runner.py, located by parsing THAT assignment.
      Not by value; not the runner file as a whole; not a fifth entry; not any
      other assignment; not any other file. Inert in `pre`, where the dict is empty;
  (c) the five pinned own records, fixed by the step-5 survey.
No exemption for tests or cards: they refer to the blocks without their numbers.

CONTROLS -- run in BOTH modes; every one must be REJECTED, and each DIFFERENTIAL
control has a baseline that must be CLEAN:
  * the survey's static set: spent H1/H2/H3/D1 blocks, H4's blocks, the TEST_ONLY
    band, the live generation range, the H4-4b mutation literal, overlap/floor;
  * A2, with the runner's SEED_BLOCKS SIMULATED FILLED in `pre` (the real file in
    `post`); baseline = filled runner alone, CLEAN -- the exemption works:
      - a bound copied into an UNRELATED ASSIGNMENT of the runner (lo: direct hit;
        hi: gap 0), plain and underscore-grouped;
      - a FIFTH SEED_BLOCKS entry repeating a block;
      - the identical SEED_BLOCKS assignment in ANOTHER file (not exempt by value);
      - a bound copied into an unrelated file (plain lo, underscore-grouped hi).
"""
import ast
import bisect
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
HERE = "docs/superpowers/evidence/t1j-h5-seed-registration"
SURVEY = "docs/superpowers/evidence/2026-09-27-t1j-h5-seed-survey/02_h5_seed_survey_run.txt"
SURVEY_SHA256 = "de7d1f693b9dadf87fca10603c0917f6ba7901ef00619e2d4fbdf4ceed5fff8f"
#: Fixed by the step-5 survey (its 02 record lists them), NOT chosen here.
OWN_RECORDS = (
    "docs/superpowers/evidence/2026-09-27-t1j-h5-seed-survey/01_h5_seed_survey.py",
    SURVEY,
    f"{HERE}/01_h5_collision_proof.py",
    f"{HERE}/02_h5_collision_proof_pre_run.txt",
    f"{HERE}/03_h5_collision_proof_post_run.txt",
)
REGISTRY_SOURCE = "scripts/GPU/alphazero/e4_screen_reference.py"
REGISTRY_NAMES = ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                  "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")
RUNNER_SOURCE = "scripts/GPU/alphazero/h5_runner.py"
SEGMENT_KEYS = (0, 1, 2, 3)
V16 = re.compile(r"(?<![\d.])(2026\d{5})(?!\d)")
GROUPED = re.compile(r"(?<![\d.])(\d(?:_?\d)*)(?![\d_])")
MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS
GEN_RANGES = [(lo, hi) for lo, hi, _why in H3R.RETIRED_GENERATION_RANGES]
GEN_RANGES.append(H3R.generation_seed_range(H3R.GEN_SEED_UNIFORM))
NAMED_SPENT = (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3,
               *SEGMENT_SEED_BLOCKS, *RETIRED_SEGMENT_BLOCKS, *R4.SEED_BLOCKS.values())
REGISTRIES = {"ACCOUNTED": [tuple(iv) for iv in ACCOUNTED_SEED_INTERVALS],
              "EXPOSED": [tuple(iv) for iv in EXPOSED_SEED_INTERVALS],
              "RETIRED": [tuple(iv) for iv in RETIRED_SEED_INTERVALS],
              "TEST_ONLY": [tuple(iv) for iv in TEST_ONLY_SEED_INTERVALS]}


# ─────────────────────────── the blocks, from the survey ───────────────────────────

def survey_blocks():
    """The four CLEAN candidates, READ from the committed survey record after its
    hash recomputes -- never retyped."""
    raw = (ROOT / SURVEY).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SURVEY_SHA256:
        raise SystemExit(f"{SURVEY} does not hash to its pin")
    text = raw.decode("utf-8")
    if "SURVEY VERDICT   : ALL FOUR CANDIDATES CLEAN" not in text:
        raise SystemExit("the survey record is not ALL FOUR CANDIDATES CLEAN")
    found = re.findall(r"^H5_STUDY_SEGMENT_(\d)\s+\((\d+), (\d+)\).*CLEAN=True$", text, re.M)
    blocks = {int(k): (int(lo), int(hi)) for k, lo, hi in found}
    assert sorted(blocks) == list(SEGMENT_KEYS) and len(found) == 4, found
    assert all(hi - lo == 2 * H5.PAIRS_PER_SEGMENT for lo, hi in blocks.values())
    return blocks


# ─────────────────────────── provenance: what is exempt ───────────────────────────

def _int_pair(node):
    return (isinstance(node, ast.Tuple) and len(node.elts) == 2
            and all(isinstance(e, ast.Constant) and type(e.value) is int for e in node.elts))


def registry_tuple_occurrences(text):
    """(a): v16's rule, unchanged."""
    out = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) in REGISTRY_NAMES for t in node.targets):
            for t in ast.walk(node.value):
                if _int_pair(t):
                    out.update((e.lineno, e.col_offset) for e in t.elts)
    return out


def seed_blocks_occurrences(text):
    """(b), AMENDMENT A2: the int elements of the (lo, hi) VALUE of each entry whose
    key is one of the four segment ints, in the module-level `SEED_BLOCKS`
    assignment -- and nothing else. A fifth key, a non-pair value or any other
    assignment exempts nothing."""
    out = set()
    for node in ast.parse(text).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if getattr(target, "id", None) != "SEED_BLOCKS" or not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Constant) and type(k.value) is int \
                    and k.value in SEGMENT_KEYS and _int_pair(v):
                out.update((e.lineno, e.col_offset) for e in v.elts)
    return out


def scan(text, exempt=frozenset()):
    """{value: count}: v16's matches UNION underscore-grouped nine-digit 2026 runs."""
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    hits, seen = {}, set()

    def keep(pos, value):
        line = bisect.bisect_right(starts, pos)
        if (line, pos - starts[line - 1]) not in exempt:
            hits[value] = hits.get(value, 0) + 1
    for m in V16.finditer(text):
        seen.add(m.start())
        keep(m.start(), int(m.group(1)))
    for m in GROUPED.finditer(text):
        digits = m.group(1).replace("_", "")
        if "_" in m.group(1) and len(digits) == 9 and digits.startswith("2026") \
                and m.start() not in seen:
            keep(m.start(), int(digits))
    return hits


def exempt_for(rel, text):
    if rel == REGISTRY_SOURCE:
        return registry_tuple_occurrences(text)
    if rel == RUNNER_SOURCE:
        return seed_blocks_occurrences(text)
    return frozenset()


def repo_sources():
    """(rel, text) for every scanned file; a path outside the repo is never exempt."""
    out = []
    for base in ("scripts", "tests", "docs/superpowers"):
        for p in (ROOT / base).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".json", ".jsonl", ".txt"):
                out.append((str(p.relative_to(ROOT)), p.read_text(errors="ignore")))
    return out


def literals_of(sources):
    vals = set()
    for rel, text in sources:
        if rel not in OWN_RECORDS:
            vals.update(scan(text, exempt_for(rel, text)))
    return vals


# ─────────────────────────────── the check ───────────────────────────────

def intervals_set(ivs):
    return {s for lo, hi in ivs for s in range(lo, hi)}


def derivations(seeds):
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


def check(block, others, literals, *, own_registration=None):
    ours = set(range(*block))
    reg = {k: [iv for iv in v if iv != own_registration] for k, v in REGISTRIES.items()}
    cats = {k: intervals_set(v) for k, v in reg.items()}
    cats.update({"CONSUMED_SEEDS": set(CONSUMED_SEEDS), "NAMED_SPENT": intervals_set(NAMED_SPENT),
                 "REPO_LITERALS": set(literals), "OTHER_CANDIDATES": intervals_set(others)})
    for i, gr in enumerate(GEN_RANGES):
        cats[f"GEN_RANGE_{i}"] = set(range(*gr))
    prior = set().union(*cats.values())
    vals = derivations(ours)
    floor = block[1] - block[0]
    ivs = ([iv for v in reg.values() for iv in v] + list(NAMED_SPENT) + list(GEN_RANGES)
           + list(others) + [(s, s + 1) for s in set(CONSUMED_SEEDS) | set(literals)])
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ivs if not (lo < block[1] and block[0] < hi)]
    direct = {k: len(ours & v) for k, v in cats.items() if ours & v}
    streams = len(vals & derivations(prior))
    return {"block": block, "direct": direct, "streams": streams,
            "injective": len(vals) == len(ours) * 5, "nearest_gap": min(gaps), "floor": floor,
            "clean": not direct and not streams and len(vals) == len(ours) * 5
            and min(gaps) >= floor}


def filled_runner(text, blocks):
    """The runner as registration will write it: SEED_BLOCKS holding the blocks."""
    empty = "SEED_BLOCKS: Dict[int, tuple] = {}"
    if seed_blocks_occurrences(text):
        return text                               # post: the real, filled file
    assert text.count(empty) == 1, "the runner's SEED_BLOCKS line is not the step-3 form"
    body = ", ".join(f"{k}: ({lo:_}, {hi:_})" for k, (lo, hi) in sorted(blocks.items()))
    return text.replace(empty, f"SEED_BLOCKS: Dict[int, tuple] = {{{body}}}")


def mutation_literal():
    """The planted wrong block of the H4 4b control 'a seed block is not the
    registered one' -- READ from the control list (as the survey did)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_controls", ROOT / "scripts/GPU/alphazero/injected_defect_controls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (new,) = [d[3] for d in mod.DEFECTS if d[0] == "H4 4b: a seed block is not the registered one"]
    return [int(m.group(1).replace("_", "")) for m in GROUPED.finditer(new)
            if len(m.group(1).replace("_", "")) == 9][0]


_LINES = []


def say(*a):
    print(*a)
    _LINES.append(" ".join(str(x) for x in a))


def main(mode, out):
    if out is None or pathlib.Path(out).exists():
        raise SystemExit("--out names a create-only run record that must not exist")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                            "scripts", "tests"], capture_output=True, text=True).stdout.strip()
    blocks = survey_blocks()
    say(f"H5 STEP 6 -- COLLISION PROOF -- MODE {mode.upper()}")
    say("=" * 72)
    say(f"HEAD {head}; scripts/ and tests/ {'DIRTY' if dirty else 'clean'}")
    say(f"blocks READ from {SURVEY} (sha256 pin recomputed)")
    say(f"exemptions: registry tuples (v16); SEED_BLOCKS segment entries (A2); "
        f"{len(OWN_RECORDS)} pinned own records")
    sources = repo_sources()
    literals = literals_of(sources)
    say(f"repo literal values counted: {len(literals)}")
    # ── registry and runner state for the mode
    state_ok = True
    for k, b in sorted(blocks.items()):
        hits = {n: sum(1 for lo, hi in v if lo < b[1] and b[0] < hi)
                for n, v in REGISTRIES.items()}
        hits = {n: c for n, c in hits.items() if c}
        exact = [n for n, v in REGISTRIES.items() if b in v]
        want_hits, want_exact = ({}, []) if mode == "pre" else ({"ACCOUNTED": 1}, ["ACCOUNTED"])
        ok = hits == want_hits and exact == want_exact
        state_ok &= ok
        say(f"  registry state segment {k} {b}: overlapping {hits or 'NONE'}; exactly in "
            f"{exact or 'NONE'}  {'OK' if ok else 'WRONG'}")
    want_runner = {} if mode == "pre" else blocks
    runner_ok = dict(H5.SEED_BLOCKS) == want_runner
    state_ok &= runner_ok
    say(f"  h5_runner.SEED_BLOCKS {'empty' if not H5.SEED_BLOCKS else 'filled'}, "
        f"{'as required' if runner_ok else 'NOT as required'} for {mode}")
    say("")
    results = {}
    for k, block in sorted(blocks.items()):
        others = tuple(b for j, b in blocks.items() if j != k)
        r = results[k] = check(block, others, literals,
                               own_registration=block if mode == "post" else None)
        say(f"H5_STUDY_SEGMENT_{k} {r['block']}  direct={r['direct'] or 'NONE'}  "
            f"streams={r['streams']}  injective={r['injective']}  "
            f"gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    say("")
    say("NEGATIVE CONTROLS -- each must be REJECTED")
    say("-" * 72)
    allb = tuple(blocks.values())
    b0 = blocks[0]
    others0 = tuple(b for j, b in blocks.items() if j != 0)
    own0 = b0 if mode == "post" else None
    rejected, total = 0, 0

    def verdict(label, ok, detail=""):
        nonlocal rejected, total
        total += 1
        rejected += ok
        say(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {label}{detail}")

    static = [("H4 study segment 0", R4.SEED_BLOCKS[("study", 0)]),
              ("H4 pilot block", R4.SEED_BLOCKS[("pilot", 0)]),
              ("H3 segment-0 replacement block", SEGMENT_SEED_BLOCKS[0]),
              ("H3's retired segment-0 quarter", RETIRED_SEGMENT_BLOCKS[0]),
              ("the H3 pilot block", H3_PILOT), ("H2 attempt-3 block", H2_A3),
              ("D1's paper reservation", D1_PAPER),
              ("the TEST_ONLY band", REGISTRIES["TEST_ONLY"][0]),
              ("inside the live H3 generation range", (GEN_RANGES[-1][0], GEN_RANGES[-1][0] + 148)),
              ("overlaps segment 0 by one seed", (b0[1] - 1, b0[1] + 147)),
              ("starts where segment 0 ends", (b0[1], b0[1] + 148)),
              ("one seed inside segment 0's floor", (b0[1] + 147, b0[1] + 295))]
    mut = mutation_literal()
    static.append((f"the H4-4b MUTATION literal {mut} (read from the control list)",
                   (mut, mut + 148)))
    for label, block in static:
        verdict(label, not check(block, allb, literals)["clean"], f"  {block}")

    # ── A2 DIFFERENTIAL controls on the runner, SIMULATED filled in `pre`
    runner_text = dict(sources)[RUNNER_SOURCE]
    filled = filled_runner(runner_text, blocks)
    rest = [(rel, t) for rel, t in sources if rel != RUNNER_SOURCE]

    def with_runner(text, extra=()):
        return literals_of(rest + [(RUNNER_SOURCE, text)] + list(extra))
    base = check(b0, others0, with_runner(filled), own_registration=own0)
    verdict("BASELINE: the filled runner alone is CLEAN (the A2 exemption works)",
            base["clean"], f"  direct={base['direct'] or 'NONE'} gap={base['nearest_gap']}")
    lo, hi = b0
    lo_hit = lambda r: r["direct"].get("REPO_LITERALS", 0) >= 1          # noqa: E731
    hi_gap = lambda r: not r["direct"] and r["nearest_gap"] == 0 < r["floor"]   # noqa: E731
    fifth = filled.replace("SEED_BLOCKS: Dict[int, tuple] = {",
                           f"SEED_BLOCKS: Dict[int, tuple] = {{4: ({lo}, {hi}), ", 1)
    cases = [
        ("lo copied into an UNRELATED ASSIGNMENT of the runner (plain)",
         with_runner(filled + f"\nOTHER_BLOCK = ({lo}, 1)\n"), lo_hit),
        ("lo copied into an UNRELATED ASSIGNMENT of the runner (underscore-grouped)",
         with_runner(filled + f"\nOTHER_BLOCK = ({lo:_}, 1)\n"), lo_hit),
        ("hi copied into an UNRELATED ASSIGNMENT of the runner (underscore-grouped)",
         with_runner(filled + f"\nOTHER_END = {hi:_}\n"), hi_gap),
        ("a FIFTH SEED_BLOCKS entry repeating segment 0", with_runner(fifth), lo_hit),
        ("the identical SEED_BLOCKS assignment in ANOTHER file",
         with_runner(filled, [("scripts/GPU/alphazero/elsewhere.py",
                               "SEED_BLOCKS = {0: (%d, %d)}\n" % (lo, hi))]), lo_hit),
        ("lo copied into an UNRELATED FILE (plain)",
         with_runner(filled, [("docs/superpowers/unrelated_notes.md", f"a number: {lo}\n")]),
         lo_hit),
        ("hi copied into an UNRELATED FILE (underscore-grouped)",
         with_runner(filled, [("docs/superpowers/unrelated_notes.md", f"STRAY = {hi:_}\n")]),
         hi_gap),
    ]
    for label, lits, failed_on in cases:
        r = check(b0, others0, lits, own_registration=own0)
        verdict(label, base["clean"] and not r["clean"] and failed_on(r),
                f"  direct={r['direct'] or 'NONE'} gap={r['nearest_gap']}")
    say(f"\ncontrols rejected: {rejected}/{total}")
    clean = all(r["clean"] for r in results.values()) and state_ok and rejected == total \
        and not dirty
    say(f"VERDICT          : {'CLEAN' if clean else 'NOT CLEAN'}"
        f"{' -- RUN ON A DIRTY TREE' if dirty else ''}")
    with open(out, "x", encoding="utf-8") as fh:            # create-only
        fh.write("\n".join(_LINES) + "\n")
    return 0 if clean else 1


if __name__ == "__main__":
    MODE = sys.argv[1] if len(sys.argv) > 1 else ""
    if MODE not in ("pre", "post"):
        raise SystemExit("usage: pre | post --out PATH")
    OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    sys.exit(main(MODE, OUT))
