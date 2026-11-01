"""The two-band seed method (`seed_band_2027`): the scanner extended to the 2027 band and the rule that places a block
there. Synthetic values are computed from the bands' bounds, so no band literal is spelled here; the pinned survey
module is loaded as a private copy and never changed on disk."""
import hashlib
import importlib.util
import pathlib
import random

import pytest

from scripts.GPU.alphazero import seed_band_2027 as B

ROOT = pathlib.Path(__file__).resolve().parents[1]
PINNED = ROOT / "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py"
PINNED_SHA256 = "6f6e55dbd5da672848c757753d4e61a3e9d110d96569b7ef468f902cf46f8860"
OLD, NEW = B.BAND_2026[0], B.BAND[0]


def _pinned(name="_tactical_survey_for_bands"):
    spec = importlib.util.spec_from_file_location(name, PINNED)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


# ─────────────────────────────── the scanner ───────────────────────────────

@pytest.mark.parametrize("value", [OLD, OLD + 54_321, B.BAND_2026[1] - 1, NEW, NEW + 12_345, B.BAND[1] - 1])
@pytest.mark.parametrize("form", ["plain", "grouped"])
def test_BOTH_BANDS_ARE_SEEN_plain_and_underscore_grouped(value, form):
    text = f"x = {value}\n" if form == "plain" else f"x = {value:_}\n"
    assert B.scan(text) == {value: 1}


@pytest.mark.parametrize("text", [f"a = {OLD - 1}\n", f"b = {B.BAND[1]}\n", f"c = {NEW + 5}0\n", f"d = 1{NEW + 5}\n",
                                  f"e = 0.{NEW + 5}\n", f"f = {OLD - 1:_}\n", f"g = {B.BAND[1]:_}\n",
                                  f"h = {NEW + 5:_}_1\n"],
                         ids=["2025-prefix", "2028-prefix", "ten-digits", "digit-before", "decimal", "2025-grouped",
                              "2028-grouped", "grouped-trailing-underscore-digit"])
def test_NEITHER_BAND_S_NEIGHBOURS_ARE_COUNTED(text):
    assert B.scan(text) == {}


def test_EVERY_OCCURRENCE_IS_COUNTED_and_the_exemption_is_by_line_and_column():
    text = f"x = {NEW + 1}, {NEW + 1:_}\ny = ({OLD + 2}, {NEW + 1})\n"
    assert B.scan(text) == {NEW + 1: 3, OLD + 2: 1}
    col = text.splitlines()[1].index(str(NEW + 1))
    assert B.scan(text, frozenset({(2, col)})) == {NEW + 1: 2, OLD + 2: 1}


def test_ON_THE_2026_BAND_IT_IS_THE_PINNED_SCANNER_EXACTLY():
    """Differential against the pinned `scan` on random texts holding only 2026-band (and out-of-band) numbers: every
    count and every exemption agrees."""
    S = _pinned()
    rng = random.Random(7)
    for _ in range(300):
        parts = []
        for _ in range(rng.randrange(1, 8)):
            v = rng.choice([OLD + rng.randrange(10 ** 5), OLD - 1 - rng.randrange(10 ** 4), rng.randrange(10 ** 9)])
            parts.append(rng.choice([f"{v}", f"{v:_}", f"x{v}", f".{v}", f"{v}0", f"{v} ", f"[{v}]"]))
        text = rng.choice([" ", "\n", ", "]).join(parts) + "\n"
        text = text.replace(str(NEW)[:4], str(OLD)[:4])         # no 2027-prefixed run: the pinned band only
        assert B.scan(text) == S.scan(text), text
        assert B.scan(text, grouped=False) == S.scan(text, grouped=False), text


def test_THE_PINNED_SCANNER_ALONE_MISSES_THE_2027_BAND():
    S = _pinned()
    assert S.scan(f"x = {NEW + 3}\n") == {} and B.scan(f"x = {NEW + 3}\n") == {NEW + 3: 1}


def test_INSTALL_REBINDS_ONLY_THE_PINNED_MODULE_S_SCANNER_and_its_repo_literals_follow(tmp_path):
    S = _pinned("_tactical_survey_install")
    before = {k: v for k, v in vars(S).items() if k != "scan"}
    (tmp_path / "notes.py").write_text(f"STRAY = {NEW + 9:_}\nOLD = {OLD + 9}\n")
    assert S.repo_literals([tmp_path / "notes.py"], (0, 0), tmp_path) == {OLD + 9: [("notes.py", 1)]}
    B.install(S)
    B.install(S)                                             # idempotent
    assert S.scan is B.scan and {k: v for k, v in vars(S).items() if k != "scan"} == before
    assert S.repo_literals([tmp_path / "notes.py"], (0, 0), tmp_path) == {NEW + 9: [("notes.py", 1)],
                                                                         OLD + 9: [("notes.py", 1)]}
    assert _pinned("_tactical_survey_fresh").scan is not B.scan    # a private copy: nothing else changed
    assert hashlib.sha256(PINNED.read_bytes()).hexdigest() == PINNED_SHA256


def test_INSTALL_REFUSES_A_MODULE_WHOSE_SCANNER_IS_NOT_THE_PINNED_ONE():
    S = _pinned("_tactical_survey_wrong")
    S.BAND = B.BAND
    with pytest.raises(ValueError, match="not the pinned survey module"):
        B.install(S)


# ─────────────────────────────── the rule ───────────────────────────────

def test_AN_EMPTY_BAND_starts_N_above_its_lower_edge():
    for n in (1, 32, 144, 1000):
        anchor, base, block = B.candidate([], [], n)
        want = -(-(NEW + n) // B.ALIGN) * B.ALIGN
        assert (anchor, base, block) == (None, want, (want, want + n)) and base - NEW >= n


def test_THE_2026_BAND_IS_IGNORED_by_the_rule_but_its_top_stays_at_least_N_below():
    ivs = [(OLD + 10, OLD + 20), (B.BAND_2026[1] - 50, B.BAND_2026[1])]
    lits = {B.BAND_2026[1] - 1: [("x", 1)]}
    anchor, base, block = B.candidate(ivs, lits, 1000)
    assert anchor is None and block == B.candidate([], [], 1000)[2] and base - B.BAND_2026[1] >= 1000


@pytest.mark.parametrize("ivs, lits, anchor", [
    ([(NEW + 10, NEW + 42)], [], NEW + 41),                              # an interval: its last seed
    ([], [NEW + 5_000], NEW + 5_000),                                    # a literal
    ([(NEW + 10, NEW + 42)], [NEW + 7, NEW + 9_999], NEW + 9_999),       # the highest of both
    ([(NEW + 9_000, NEW + 9_500), (NEW + 100, NEW + 200)], [NEW + 300], NEW + 9_499),
    ([(OLD + 5, NEW + 30)], [], NEW + 29)],                              # straddling the edge: its last seed counts
    ids=["interval", "literal", "both", "unordered", "straddling"])
def test_THE_BLOCK_LIES_ABOVE_EVERY_INTERVAL_AND_LITERAL_IN_THE_BAND_with_an_N_seed_margin(ivs, lits, anchor):
    got_anchor, base, block = B.candidate(ivs, lits, 1000)
    assert got_anchor == anchor and base % B.ALIGN == 0
    assert base - (anchor + 1) >= 1000 and base - B.ALIGN - (anchor + 1) < 1000     # the margin, and the LEAST such
    assert all(hi <= block[0] for _lo, hi in ivs) and all(v < block[0] for v in lits)


def test_THE_LITERALS_MAY_BE_THE_SCAN_S_MAPPING_and_the_input_is_not_changed():
    lits = {NEW + 777: [("a.py", 2)], OLD + 1: [("b.py", 1)]}
    ivs = [(NEW + 10, NEW + 20)]
    before = (dict(lits), list(ivs))
    assert B.candidate(ivs, lits, 32)[0] == NEW + 777 and (lits, ivs) == before


def test_IT_REFUSES_once_the_block_would_leave_the_band_and_never_wraps():
    assert B.candidate([], [B.BAND[1] - 2_101], 1000)[2][1] <= B.BAND[1]          # control: still fits
    with pytest.raises(B.Exhausted, match="would leave the 2027 band"):
        B.candidate([], [B.BAND[1] - 1_000], 1000)
    with pytest.raises(B.Exhausted):
        B.candidate([(B.BAND[1] + 5, B.BAND[1] + 9)], [], 1)                    # an interval above the band refuses


@pytest.mark.parametrize("bad", [0, -3, 2.0, True, None])
def test_A_BAD_BLOCK_SIZE_IS_REFUSED(bad):
    with pytest.raises(ValueError, match="positive int"):
        B.candidate([], [], bad)


# ─────────────────────────────── the check, across the bands ───────────────────────────────

def _prior(literals=None, registries=None):
    """Prior terms with one far-away registered seed (the check needs at least one other interval for its gap)."""
    return ({"ACCOUNTED": [], "EXPOSED": [], "RETIRED": [], "TEST_ONLY": [(OLD + 90_000, OLD + 90_001)],
             **(registries or {})}, {}, [], set(), literals or {})


def test_A_CROSS_BAND_DERIVED_STREAM_COLLISION_IS_REJECTED_both_ways():
    """The pinned check is band-agnostic: with a (synthetic) mask that takes a 2027-band seed onto a 2026-band literal
    or registered seed, the block is rejected on its streams; without the plant it is CLEAN."""
    S = _pinned()
    block = B.candidate([], [], 1000)[2]
    v = OLD + 54_321
    m = block[0] ^ v
    assert S.check(block, _prior(), [m])["streams"] == 0
    assert S.check(block, _prior({v: [("x", 1)]}), [m])["streams"] >= 1
    assert S.check(block, _prior(registries={"EXPOSED": [(v, v + 1)]}), [m])["streams"] >= 1
    mine = (OLD + 10_000, OLD + 11_000)                       # and a 2026-band block onto a 2027-band literal
    w = NEW + 4_321
    assert S.check(mine, _prior({w: [("x", 1)]}), [mine[0] ^ w])["streams"] >= 1


def test_WITH_THE_REAL_MASKS_NO_2026_VALUE_SHARES_A_DERIVED_STREAM_WITH_A_2027_VALUE():
    """So a cross-band stream collision is impossible with the four masks in use; the check would catch one anyway."""
    masks = _pinned().prior_terms()[3]
    assert len(masks) == 4

    def der(rng):
        return {x for s in rng for x in (s, *(s ^ k for k in masks))}
    assert not der(range(*B.BAND_2026)) & der(range(*B.BAND))


def test_THE_GAP_FLOOR_IS_THE_BLOCK_S_SIZE_against_literals_in_either_band():
    S = _pinned()
    masks = S.prior_terms()[3]
    block = B.candidate([], [], 1000)[2]
    assert S.check(block, _prior(), masks)["floor"] == 1000 and S.check(block, _prior(), masks)["clean"]
    near = block[0] - 1000                                    # exactly the floor below: clean; one closer: not
    assert S.check(block, _prior({near - 1: [("x", 1)]}), masks)["clean"]
    assert not S.check(block, _prior({near: [("x", 1)]}), masks)["clean"]
    assert not S.check(block, _prior({block[1] + 999: [("x", 1)]}), masks)["clean"]
    assert S.check(block, _prior({block[1] + 1000: [("x", 1)]}), masks)["clean"]


# ─────────────────────────────── the real state ───────────────────────────────

def test_REAL_STATE_the_2027_band_holds_generation_s_closed_block_and_the_screen_s_ACCOUNTED_block():
    """RE-ANCHORED 2026-10-30 (registration 326004d0): the band's first and only registered block is generation's,
    read from its committed plan (no literal here), ACCOUNTED only; the rule would now place a later block above it.
    RE-ANCHORED 2026-10-30 (generation's closing, K2 569ba2a4): that block is now also EXPOSED and RETIRED whole, the
    band's only entry in each; nothing in the band is TEST_ONLY.
    RE-ANCHORED 2026-10-31 (the screen's registration 3aaff337): the screen's block, read from its committed plan, is
    the band's second ACCOUNTED entry, above generation's, and in no other list; generation's is still the band's only
    EXPOSED and RETIRED entry; the rule would now place a later block above the screen's."""
    import json
    from scripts.GPU.alphazero import e4_screen_reference as REG
    blocks = []
    for kind in ("generate", "screen"):
        plan = json.loads((ROOT / f"docs/superpowers/evidence/combination-policy-{kind}-run/plan.json").read_text())
        blocks.append((plan["base"], plan["base"] + plan["n_slots"]))
    gen, screen = blocks
    assert gen[1] <= screen[0]
    assert [iv for iv in REG.ACCOUNTED_SEED_INTERVALS if iv[1] > NEW] == [gen, screen]
    for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
        assert [iv for iv in getattr(REG, name) if iv[1] > NEW] == [gen], name
    assert all(hi <= NEW for _lo, hi in REG.TEST_ONLY_SEED_INTERVALS)
    assert B.candidate(REG.ACCOUNTED_SEED_INTERVALS, [], 64)[2][0] >= screen[1] + 64


def test_THIS_MODULE_HOLDS_NO_SEED_LITERAL():
    import re
    text = (ROOT / "scripts/GPU/alphazero/seed_band_2027.py").read_text()
    assert not B.scan(text) and not re.search(r"\b20[0-9]{7}\b", text)
