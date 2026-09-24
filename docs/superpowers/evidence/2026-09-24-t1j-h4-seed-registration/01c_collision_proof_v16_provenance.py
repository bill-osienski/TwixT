"""H4 SEED REGISTRATION -- COLLISION PROOF v16, PROVENANCE (step-4 card amendment 7).

Supersedes `01b` (kept unchanged; it produced the CLEAN pre run 02b and the NOT
CLEAN post run 03). Amendment 7's changes, and nothing else:
  * a literal is exempt ONLY at its specific occurrence as an element of a
    (lo, hi) tuple inside one of the four registry assignments of
    e4_screen_reference.py, located by PARSING that file; the same number anywhere
    else counts -- no value is exempt globally (01b dropped every value inside
    ACCOUNTED, so after registration a copied candidate seed was ignored);
  * ten pinned own records (amendment 6's seven + this script + 03 + 03b);
  * two DIFFERENTIAL controls in both modes: a candidate seed, and an exclusive
    end, each copied into an unrelated real file, must fail the block on the
    literal itself / on the gap to it, while the same block without the copy is
    CLEAN.

--- the 01b docstring follows ---

The original `01_collision_proof_v16.py` is kept unchanged: it produced the
NOT CLEAN pre-registration run (02a), whose only literal hits came from this
reservation's own records. This copy differs in exactly three ways:
  1. the repo-literal term skips SEVEN EXACTLY PINNED PATHS (the card, the survey
     script and run, both v16 scripts, the NOT CLEAN run, the re-run) -- every
     other file's literals still count;
  2. a 14th negative control copies a candidate literal into an UNRELATED real
     file and requires the block to be rejected;
  3. `--out PATH` writes the run create-only (the file may not exist).

    python -m <this file as a path>  pre    # before registration: MUST be CLEAN
    python -m <this file as a path>  post   # after: each block excluded BY IDENTITY

The five H4 blocks of the step-4 card §2 (bound card), proved with v15's method
unchanged -- each candidate seed and its four XOR-mask derivations disjoint from
every prior term (all four registries, CONSUMED_SEEDS, the named spent H1/H2/H3/D1
blocks, the four H3 generation ranges no registry holds); derivations injective;
nearest other interval at least the candidate's OWN size away -- plus the two
additions of the 2026-09-24 survey: the five candidates are priors for EACH OTHER,
and every nine-digit 2026 literal in scripts/tests/docs outside a registered
interval is a prior term.

`pre` runs against the registries AS THEY STAND BEFORE REGISTRATION (step-4
card amendment 5): the blocks must be absent from every registry and CLEAN.
`post` runs after they are written into ACCOUNTED_SEED_INTERVALS: each block must
be registered as EXACTLY its own interval, neither exposed nor retired, and is
excluded from the priors BY IDENTITY for its own check -- never by subtracting its
seeds, which (2026-09-04) excused every control equal to an existing reservation.

Nothing is drawn, registered or built. Registries and masks are IMPORTED.
"""
import pathlib
import re
import sys

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

ROOT = pathlib.Path(__file__).resolve().parents[4]
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
assert MODE in ("pre", "post"), "usage: pre | post [--out PATH]"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None

#: 🔴 Step-4 card amendment 6: EXACT paths, pinned there and here. Not prefixes.
EXCLUDED_OWN_RECORDS = frozenset((
    "docs/superpowers/2026-09-24-t1j-h4-step4-seed-card.md",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-step4-collision-survey/01_collision_survey.py",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-step4-collision-survey/02_collision_survey_run.txt",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/01_collision_proof_v16.py",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/01b_collision_proof_v16_amended.py",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/02a_collision_proof_v16_pre_run_NOT_CLEAN.txt",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/02b_collision_proof_v16_pre_run.txt",
    # amendment 7
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/01c_collision_proof_v16_provenance.py",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/03_collision_proof_v16_post_run.txt",
    "docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/03b_collision_proof_v16_post_run.txt",
))
assert len(EXCLUDED_OWN_RECORDS) == 10
REGISTRY_SOURCE = "scripts/GPU/alphazero/e4_screen_reference.py"
REGISTRY_NAMES = ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                  "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS")


def registry_tuple_occurrences(text):
    """(line, col) of every int that is an element of a (lo, hi) tuple inside one
    of the four registry assignments -- by PARSING, never by value."""
    import ast
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

#: The step-4 card §2 table, transcribed HERE -- the proof checks the card's
#: blocks, not whatever the code later calls them.
BLOCKS = {"H4_PILOT": (202_632_000, 202_632_032),
          "H4_STUDY_SEGMENT_0": (202_634_000, 202_634_148),
          "H4_STUDY_SEGMENT_1": (202_636_000, 202_636_148),
          "H4_STUDY_SEGMENT_2": (202_638_000, 202_638_148),
          "H4_STUDY_SEGMENT_3": (202_640_000, 202_640_148)}
assert [hi - lo for lo, hi in BLOCKS.values()] == [32, 148, 148, 148, 148]

MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS
GEN_RANGES = [(lo, hi) for lo, hi, _why in H3R.RETIRED_GENERATION_RANGES]
GEN_RANGES.append(H3R.generation_seed_range(H3R.GEN_SEED_UNIFORM))
NAMED_SPENT = (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3,
               *SEGMENT_SEED_BLOCKS, *RETIRED_SEGMENT_BLOCKS)


def repo_files():
    return [p for base in ("scripts", "tests", "docs/superpowers")
            for p in (ROOT / base).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md", ".json", ".jsonl", ".txt")]


def repo_literals(files=None):
    """2026 literals by PROVENANCE: every occurrence counts, in every file, except
    one inside a pinned own record or one that IS a registry-tuple element."""
    out = set()
    for p in (repo_files() if files is None else files):
        try:
            rel = str(p.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(p)                          # outside the repo: never excluded
        if rel in EXCLUDED_OWN_RECORDS:
            continue
        text = p.read_text(errors="ignore")
        exempt = registry_tuple_occurrences(text) if rel == REGISTRY_SOURCE else set()
        starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
        for m in re.finditer(r"(?<![\d.])(2026\d{5})(?!\d)", text):
            import bisect
            line = bisect.bisect_right(starts, m.start())
            if (line, m.start() - starts[line - 1]) in exempt:
                continue
            out.add(int(m.group(1)))
    return out


def intervals_set(ivs):
    return {s for lo, hi in ivs for s in range(lo, hi)}


REGISTRIES = {"ACCOUNTED": list(ACCOUNTED_SEED_INTERVALS),
              "EXPOSED": list(EXPOSED_SEED_INTERVALS),
              "RETIRED": list(RETIRED_SEED_INTERVALS),
              "TEST_ONLY": list(TEST_ONLY_SEED_INTERVALS)}
LITERALS = repo_literals()


def derivations(seeds):
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


def check(block, others=(), *, own_registration=None):
    """`own_registration`: the interval this block is itself registered as,
    excluded BY IDENTITY from the registries and from the gap computation."""
    ours = set(range(*block))
    reg = {k: [iv for iv in v if tuple(iv) != own_registration] for k, v in REGISTRIES.items()}
    cats = {k: intervals_set(v) for k, v in reg.items()}
    cats.update({"CONSUMED_SEEDS": set(CONSUMED_SEEDS),
                 "NAMED_SPENT": intervals_set(NAMED_SPENT),
                 "REPO_LITERALS": set(LITERALS),
                 "OTHER_CANDIDATES": intervals_set(others)})
    for i, gr in enumerate(GEN_RANGES):
        cats[f"GEN_RANGE_{i}"] = set(range(*gr))
    prior = set().union(*cats.values())
    vals = derivations(ours)
    floor = block[1] - block[0]
    ivs = ([tuple(iv) for v in reg.values() for iv in v] + list(NAMED_SPENT)
           + list(GEN_RANGES) + list(others)
           + [(s, s + 1) for s in set(CONSUMED_SEEDS) | LITERALS])
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ivs if not (lo < block[1] and block[0] < hi)]
    nearest = min(gaps)
    direct = {k: len(ours & v) for k, v in cats.items() if ours & v}
    streams = len(vals & derivations(prior))
    return {"block": block, "n": len(ours), "direct": direct, "streams": streams,
            "injective": len(vals) == len(ours) * 5, "nearest_gap": nearest,
            "floor": floor, "clean": not direct and not streams
            and len(vals) == len(ours) * 5 and nearest >= floor}


_LINES = []
_print = print


def print(*a, **k):                               # noqa: A001 -- tee to the create-only record
    _print(*a, **k)
    _LINES.append(" ".join(str(x) for x in a))


if __name__ == "__main__":
    if OUT is not None and pathlib.Path(OUT).exists():
        raise SystemExit(f"{OUT} exists: the run record is create-only")
    missing = [p for p in EXCLUDED_OWN_RECORDS if not (ROOT / p).exists()
               and not p.endswith("03b_collision_proof_v16_post_run.txt")]
    assert not missing, f"a pinned exclusion does not exist: {missing}"
    print(f"H4 SEED REGISTRATION -- COLLISION PROOF v16 (PROVENANCE) -- MODE {MODE.upper()}")
    print(f"excluded own records (exact paths, card amendments 6-7): {len(EXCLUDED_OWN_RECORDS)}")
    print("=" * 70)
    print(f"masks (imported): {[hex(m) for m in MASKS]}")
    print(f"registries: " + ", ".join(f"{k} {len(v)} intervals" for k, v in REGISTRIES.items()))
    print(f"repo literal values counted (by provenance, registry-tuple occurrences exempt): {len(LITERALS)}")
    registered = {name: [k for k, v in REGISTRIES.items() if b in [tuple(x) for x in v]]
                  for name, b in BLOCKS.items()}
    status_ok = True
    for name, b in BLOCKS.items():
        overlapping = {k: sum(1 for lo, hi in v if lo < b[1] and b[0] < hi)
                       for k, v in REGISTRIES.items()}
        overlapping = {k: n for k, n in overlapping.items() if n}
        want = {} if MODE == "pre" else {"ACCOUNTED": 1}
        exact = registered[name] == ([] if MODE == "pre" else ["ACCOUNTED"])
        ok = overlapping == want and exact
        status_ok &= ok
        print(f"  registry state {name:20s} {b}: overlapping {overlapping or 'NONE'}"
              f"{'' if MODE == 'pre' else f', registered as exactly itself in {registered[name]}'}"
              f"  {'OK' if ok else 'WRONG'}")
    print()
    results = {}
    for name, block in BLOCKS.items():
        others = tuple(b for n, b in BLOCKS.items() if n != name)
        r = results[name] = check(block, others,
                                  own_registration=block if MODE == "post" else None)
        print(f"{name:20s} {r['block']}  n={r['n']:>3}  direct={r['direct'] or 'NONE'}  "
              f"streams={r['streams']}  injective={r['injective']}  "
              f"gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    print(f"\ntotal seeds: {sum(r['n'] for r in results.values())}")
    print("\nNEGATIVE CONTROLS -- each must be REJECTED")
    print("-" * 70)
    seg0 = BLOCKS["H4_STUDY_SEGMENT_0"]
    everything = tuple(BLOCKS.values())
    controls = [
        ("H3 segment-0 replacement block (spent)", SEGMENT_SEED_BLOCKS[0]),
        ("H3's retired segment-0 quarter", RETIRED_SEGMENT_BLOCKS[0]),
        ("the H3 pilot block", H3_PILOT),
        ("H2 attempt-3 block", H2_A3),
        ("D1's paper reservation", D1_PAPER),
        ("the TEST_ONLY band", tuple(TEST_ONLY_SEED_INTERVALS[0])),
        ("the stray literal 202630000", (202_630_000, 202_630_032)),
        ("the stray literal 202699000", (202_699_000, 202_699_032)),
        ("overlaps H4 segment 0 by one seed", (seg0[1] - 1, seg0[1] + 147)),
        ("starts where H4 segment 0 ends -- inside the floor", (seg0[1], seg0[1] + 148)),
        ("one seed inside segment 0's floor", (seg0[1] + 147, seg0[1] + 295)),
        ("EQUALS the H4 pilot block (an existing reservation, not its own)",
         BLOCKS["H4_PILOT"]),
        ("inside the live H3 uniform generation range",
         (GEN_RANGES[-1][0], GEN_RANGES[-1][0] + 148)),
    ]
    rejected = 0
    for why, block in controls:
        r = check(block, everything)
        ok = not r["clean"]
        rejected += ok
        print(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {why}  {block}")
    # 🔴 AMENDMENT 7: two DIFFERENTIAL controls. The same block, with and without
    # a literal copied into an UNRELATED real file: without it the block must be
    # CLEAN; with it, it must fail ON THAT LITERAL -- a direct REPO_LITERALS hit
    # for a copied seed, a gap of 0 to the copy for a copied exclusive end.
    import tempfile
    pilot = BLOCKS["H4_PILOT"]
    others = tuple(b for n, b in BLOCKS.items() if n != "H4_PILOT")
    own = pilot if MODE == "post" else None
    base = check(pilot, others, own_registration=own)
    for label, value, failed_on in (
            ("a candidate SEED copied into an UNRELATED file", pilot[0],
             lambda r: r["direct"].get("REPO_LITERALS", 0) >= 1),
            ("an exclusive END copied into an UNRELATED file", pilot[1],
             lambda r: not r["direct"] and r["nearest_gap"] == 0 < r["floor"])):
        with tempfile.TemporaryDirectory() as tmp:
            unrelated = pathlib.Path(tmp) / "unrelated_notes.md"
            unrelated.write_text(f"a stray number: {value}\n")
            saved = set(LITERALS)
            LITERALS.clear()
            LITERALS.update(repo_literals(repo_files() + [unrelated]))
            try:
                r = check(pilot, others, own_registration=own)
            finally:
                LITERALS.clear()
                LITERALS.update(saved)
        ok = base["clean"] and not r["clean"] and failed_on(r)
        rejected += ok
        controls.append((label, pilot))
        print(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {label} ({value}): without "
              f"the copy CLEAN={base['clean']}; with it direct={r['direct']} "
              f"gap={r['nearest_gap']}")
    print(f"\ncontrols rejected: {rejected}/{len(controls)}")
    clean = all(r["clean"] for r in results.values()) and status_ok \
        and rejected == len(controls)
    print(f"VERDICT          : {'CLEAN' if clean else 'NOT CLEAN'}")
    if OUT is not None:
        with open(OUT, "x", encoding="utf-8") as fh:        # create-only
            fh.write("\n".join(_LINES) + "\n")
    sys.exit(0 if clean else 1)
