"""The A8.8 step-4 pilot seed survey (card §18 A8.1-A8.2) -- its controls, SYNTHETIC ONLY.

The survey script is imported by path and exercised on temporary trees; it is never run on the
repository here. No band value is SPELLED in this file: every one is computed from the band's
lower end, so this file -- itself inside the survey's scope -- adds no seed literal to it.
"""
import gzip
import importlib.util
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py"


@pytest.fixture(scope="module")
def S():
    spec = importlib.util.spec_from_file_location("_pilot_seed_survey", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lo_of(S, offset=0):
    """A synthetic value INSIDE the band (which is 100,000 wide), computed: never written as a literal."""
    return S.BAND[0] + 97_310 + offset


def tree(tmp_path, files):
    for rel, raw in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw if isinstance(raw, bytes) else raw.encode())
    return tmp_path


# ─────────────────────────────── the script itself ───────────────────────────────

def test_the_script_and_this_file_SPELL_no_band_literal(S):
    for path in (SCRIPT, pathlib.Path(__file__)):
        assert S.scan(path.read_text()) == {}, path


def test_importing_the_script_SCANS_nothing(S):
    assert not hasattr(S, "LITERALS") and S._LINES == []


def test_the_frozen_names_are_the_cards(S):
    assert (S.N, S.ALIGN, S.PAIRS, S.ROOTS) == (32, 10_000, 16, ("scripts", "tests", "docs/superpowers"))
    assert S.OWN_RECORDS == (
        "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py",
        "docs/superpowers/evidence/tactical-candidate-seed-survey/02_seed_survey_run.txt",
        "docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py",
        "docs/superpowers/evidence/tactical-candidate-seed-registration/02_collision_proof_pre_run.txt",
        "docs/superpowers/evidence/tactical-candidate-seed-registration/03_collision_proof_post_run.txt")
    assert S.PLAN == "docs/superpowers/evidence/tactical-candidate-plans/pilot-s0.json"
    assert pathlib.Path(S.__file__).resolve() == SCRIPT and S.ROOT == ROOT


# ─────────────────────────────── the scope (A8.1) ───────────────────────────────

def test_the_scope_is_EVERY_regular_file_under_the_three_roots(S, tmp_path):
    t = tree(tmp_path, {"scripts/a.py": "", "tests/b": "", "docs/superpowers/evidence/x/run.log": "",
                        "docs/superpowers/c.sha256": "", "docs/superpowers/d.csv": "", "docs/other/e.py": "",
                        "notes.txt": "", "scripts/__pycache__/a.cpython.pyc": b"\x00\x01"})
    os.symlink(t / "scripts/a.py", t / "scripts/link.py")
    os.symlink(t / "docs/superpowers/evidence", t / "tests/linkdir")
    files, links = S.repo_files(t)
    assert sorted(str(p.relative_to(t)) for p in files) == sorted(
        ["scripts/a.py", "tests/b", "docs/superpowers/evidence/x/run.log", "docs/superpowers/c.sha256",
         "docs/superpowers/d.csv", "scripts/__pycache__/a.cpython.pyc"])
    assert sorted(str(p.relative_to(t)) for p in links) == ["scripts/link.py", "tests/linkdir"]


def test_GZ_is_decompressed_and_UNDECODABLE_bytes_are_REPLACED(S, tmp_path):
    v = lo_of(S)
    (tmp_path / "t.gz").write_bytes(gzip.compress(f"seed {v}\n".encode()))
    assert S.scan(S.read_text(tmp_path / "t.gz")) == {v: 1}
    (tmp_path / "b.bin").write_bytes(b"7\xff" + str(v).encode())
    assert S.scan(S.read_text(tmp_path / "b.bin")) == {v: 1}, "replaced: no digit joins the literal"
    assert S.scan(S.read_text(tmp_path / "b.bin", errors="ignore")) == {}, "dropping the byte would hide it"


@pytest.mark.parametrize("name,raw", [("locked.txt", b"x"), ("broken.gz", b"not gzip"), ("cut.gz", None)],
                         ids=["unreadable", "corrupt gzip", "truncated gzip"])
def test_an_UNREADABLE_file_or_a_FAILED_decompression_STOPS(S, tmp_path, name, raw):
    p = tmp_path / name
    p.write_bytes(raw if raw is not None else gzip.compress(b"x" * 1000)[:-8])
    if name == "locked.txt":
        p.chmod(0)
    try:
        with pytest.raises(S.SurveyStop):
            S.read_text(p)
        with pytest.raises(S.SurveyStop):
            S.repo_literals([p], (0, 1), tmp_path)
    finally:
        p.chmod(0o600)


# ─────────────────────────────── the scanner ───────────────────────────────

def test_the_UNION_scanner_sees_plain_and_underscore_forms_and_nothing_else(S):
    v = lo_of(S)
    text = (f"a = {v}\nb = {v + 1:_}\nc = 1{v}\nd = {v}9\ne = 0.{v}\nf = {str(v)[:3]}_{str(v)[3:]}\n"
            f"g = {v + 2}_\nh = '{v + 3}'\n")
    hits = S.scan(text)
    assert hits == {v: 2, v + 1: 1, v + 2: 1, v + 3: 1}, hits   # c, d: ten digits; e: a decimal
    assert S.scan(text, grouped=False) == {v: 1, v + 2: 1, v + 3: 1}, "v16 alone misses the grouped forms"


def test_REGISTRY_tuples_are_exempt_by_PARSING_even_after_a_non_ASCII_char(S):
    v = lo_of(S)
    text = (f'ACCOUNTED_SEED_INTERVALS = (("é" * 0) or ({v}, {v + 32}),)\n'
            f"RETIRED_SEED_INTERVALS = (\n    ({v + 100}, {v + 132}),  # ✓\n)\n"
            f"OTHER = ({v + 200}, {v + 232})\nNOTE = {v}\n")
    hits = S.scan(text, S.registry_tuple_occurrences(text))
    assert hits == {v + 200: 1, v + 232: 1, v: 1}, hits


# ─────────────────────────────── the exemptions (A8.2) ───────────────────────────────

def test_the_EXEMPTIONS_are_exact_paths_the_conditional_plan_and_registry_tuples(S, tmp_path):
    v = lo_of(S)
    block = (v, v + 32)
    t = tree(tmp_path, {S.OWN_RECORDS[0]: f"{v + 1}", S.OWN_RECORDS[4]: f"{v + 2}",
                        S.OWN_RECORDS[0] + ".bak": f"{v + 3}", f"{S.HERE}/03_extra.txt": f"{v + 4}",
                        S.PLAN: f"[{v}, {v + 31}]", S.REGISTRY_SOURCE: f"ACCOUNTED_SEED_INTERVALS = (({v + 5}, {v + 6}),)"})
    files, _ = S.repo_files(t)
    lits = S.repo_literals(files, block, t)
    assert sorted(lits) == [v + 3, v + 4], lits
    (t / S.PLAN).write_text(f"[{v}, {v + 32}]")           # one seed OUTSIDE the block: not exempt
    assert sorted(S.repo_literals(files, block, t)) == [v, v + 3, v + 4, v + 32]


def test_a_file_OUTSIDE_the_root_is_never_exempt(S, tmp_path):
    v = lo_of(S)
    outside = tmp_path / "elsewhere" / "02_seed_survey_run.txt"
    outside.parent.mkdir()
    outside.write_text(f"{v}")
    assert S.repo_literals([outside], (0, 1), tmp_path / "root") == {v: [(str(outside), 1)]}


# ─────────────────────────────── the rule and the block (A8.1) ───────────────────────────────

def test_BASE_is_the_first_10000_multiple_STRICTLY_above_the_highest_SEED(S):
    v = S.BAND[0]
    assert S.candidate([(v + 50_000, v + 50_148)])[1:] == (v + 60_000, (v + 60_000, v + 60_032))
    assert S.candidate([(v + 59_990, v + 60_000)])[1] == v + 60_000, "the last seed is end - 1"
    assert S.candidate([(v + 59_990, v + 60_001)])[1] == v + 70_000
    assert S.candidate([(v + 10, v + 20), (90_009_000, 90_009_100)])[1] == v + 10_000, "below the band: ignored"
    assert S.candidate([(v + 10, v + 20), (S.BAND[1] + 10, S.BAND[1] + 20)])[1] == v + 10_000, "above: ignored"


def test_the_PAIR_seeds_follow_the_frozen_assignment(S):
    lo = lo_of(S)
    p = S.pair_seeds((lo, lo + 32))
    assert [x["pair_id"] for x in p] == [f"pilot-{k:02d}" for k in range(16)]
    assert [(x["red"] - lo, x["black"] - lo) for x in p] == [(2 * k, 2 * k + 1) for k in range(16)]
    assert sorted(s for x in p for s in (x["red"], x["black"])) == list(range(lo, lo + 32))


MASKS = [0xA5A5A5, 0x5A5A5A, 0xC3C3C3, 0x3C3C3C]


def prior(S, *, registries=None, named=None, gen=(), consumed=(), literals=None):
    """Synthetic prior terms; one FAR interval always present, so a gap is always measured."""
    far = (lo_of(S, -10_000), lo_of(S, -9_000))
    regs = {"ACCOUNTED": [far], "EXPOSED": [], "RETIRED": [], "TEST_ONLY": []}
    for k, v in (registries or {}).items():
        regs[k] = regs[k] + list(v)
    return regs, named or {}, list(gen), set(consumed), literals or {}


def test_CHECK_a_clean_block_and_each_way_it_fails(S):
    lo = lo_of(S)
    block = (lo, lo + 32)
    assert S.check(block, prior(S), MASKS)["clean"]
    r = S.check(block, prior(S, literals={lo + 4: [("x", 1)]}), MASKS)
    assert not r["clean"] and r["direct"] == {"REPO_LITERALS": 1}
    r = S.check(block, prior(S, consumed={(lo + 2) ^ MASKS[1]}), MASKS)
    assert not r["clean"] and r["streams"] >= 1 and not r["direct"], "a derived stream collides"
    r = S.check(block, prior(S, registries={"RETIRED": [(lo + 32 + 31, lo + 32 + 40)]}), MASKS)
    assert not r["clean"] and r["nearest_gap"] == 31 < r["floor"] == 32, "inside the gap floor"
    r = S.check(block, prior(S, registries={"RETIRED": [(lo + 64, lo + 80)]}), MASKS)
    assert r["clean"] and r["nearest_gap"] == 32, "exactly the floor is clean"
    r = S.check(block, prior(S), [1, 2, 3, 4])
    assert not r["injective"] and not r["clean"], "derivations that collide within the block"
    r = S.check(block, prior(S, named={"H5": [(lo + 31, lo + 40)]}), MASKS)
    assert not r["clean"] and r["direct"] == {"NAMED H5": 1}


def test_the_RUN_record_is_CREATE_ONLY(S, tmp_path):
    out = tmp_path / "02.txt"
    out.write_text("already")
    with pytest.raises(SystemExit, match="create-only"):
        S.main(["x", "--out", str(out)])
    assert out.read_text() == "already"


def test_the_RUN_controls_all_PASS_on_a_synthetic_block(S):
    """The script's own run-time controls, on a synthetic block and synthetic prior terms: every
    differential, fail-closed and exemption control passes -- and none passes on an unclean block."""
    lo = lo_of(S)
    block = (lo, lo + 32)
    plants = S._plant_controls(block, prior(S), MASKS, {}, True)
    assert len(plants) == 8 and all(ok for _l, ok, _d in plants), [(l, d) for l, ok, d in plants if not ok]
    assert any("would MISS it" in d for _l, _ok, d in plants), "the replaced-byte control shows what ignore loses"
    assert not any(ok for _l, ok, _d in S._plant_controls(block, prior(S), MASKS, {}, False)[:6])
    exemptions = S._exemption_controls(block)
    assert len(exemptions) == 6 and all(ok for _l, ok, _d in exemptions), exemptions
