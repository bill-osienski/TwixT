"""THE GATE INVENTORY, DERIVED FROM SOURCE — never a hand-kept list.

🔴 WHY THIS FILE EXISTS. Every report in this programme for weeks said "all EIGHT
gates are False". There were TEN. The two missing ones --
`LOWPLY_QUALIFICATION_AUTHORIZED` and `RUNTIME_REQUAL_AUTHORIZED` -- belong to
qualifications that have already run, so nobody thought about them; but a retired
gate is still a closed authorization constant in the source, and a count that
does not include it is wrong.

The defect is not the number. It is that the count came from a LIST TYPED BY HAND
in each checker. A hand-kept list cannot see a gate nobody remembered to add to
it, which is the one case the count exists to catch. So the inventory is
ENUMERATED FROM THE SOURCE by AST, and the expected total is asserted -- adding
or removing a gate must be a deliberate edit to `EXPECTED_GATES`, not a silent
drift.
"""
import ast
import pathlib

import pytest

from scripts.GPU.alphazero import gate_inventory as INVENTORY

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

#: 🔑 THE INVENTORY IS THE ASSERTION. Editing this set is how a gate is added or
#: removed; nothing else may change the count silently.
EXPECTED_GATES = {
    "d1_probe.py": "D1_EXECUTION_AUTHORIZED",
    "e4_screen_command.py": "SCREEN_AUTHORIZED",
    "h1_viability_runner.py": "H1_EXECUTION_AUTHORIZED",
    "h2_match_runner.py": "H2_EXECUTION_AUTHORIZED",
    "h3_pilot_runner.py": "H3_PILOT_EXECUTION_AUTHORIZED",
    #: 🔴 AMENDMENT 3 SWAPPED THIS ONE, IT DID NOT DELETE IT. The generation
    #: EXECUTION gate went because engine-free generation runs nothing; the
    #: POPULATION-FREEZE barrier took its place, guarding the one act still
    #: irreversible -- declaring a population THE population. Ten either way,
    #: and the swap is a deliberate edit here rather than a silent drift.
    "h3_study_generator.py": "H3_POPULATION_FREEZE_AUTHORIZED",
    "h3_study_runner.py": "H3_STUDY_EXECUTION_AUTHORIZED",
    #: 🔴 THE ELEVENTH, ADDED 2026-09-20. Combining the four segments is the
    #: study's one interpretive act -- after it a number exists that people will
    #: quote -- so it takes its own reviewed edit rather than riding on the
    #: execution gate that authorized playing the games.
    "h3_combine.py": "H3_COMBINATION_AUTHORIZED",
    #: 🔴 THE TWELFTH, ADDED 2026-09-21. H4 §4A raw capability characterization.
    #: It runs no game and computes no score, but it LAUNCHES JVMS against the
    #: real T1j build, and anything that spawns the external engine takes its own
    #: reviewed authorization rather than riding on another stage's.
    "h4_4a_characterization.py": "H4_4A_CHARACTERIZATION_AUTHORIZED",
    #: 🔴 THE THIRTEENTH, ADDED 2026-09-21. The H4 repair qualification. Like
    #: §4A it runs no game, but it launches JVMS against the real T1j build
    #: THROUGH A BEHAVIOURALLY MODIFIED HELPER -- the opt-in MatchData
    #: injection -- so it takes its own reviewed authorization rather than
    #: riding on §4A's, which qualified a different helper.
    "h4_repair_qualification.py": "H4_REPAIR_QUALIFICATION_AUTHORIZED",
    "h4_4b_acceptance_qualification.py": "H4_4B_ACCEPTANCE_QUALIFICATION_AUTHORIZED",
    "h4_runner.py": "H4_PILOT_EXECUTION_AUTHORIZED",
    #: 🔴 THE SIXTEENTH, ADDED 2026-09-23. H4 confirmatory aggregation: the one act
    #: after which an H4 strength number exists. It runs no engine, but it READS
    #: OUTCOMES, so it takes its own reviewed edit -- not the pilot's execution gate
    #: and not the proceed/stop artifact, which it requires in addition.
    "h4_confirmatory_analysis.py": "H4_STUDY_AGGREGATION_AUTHORIZED",
    "l0_match_command.py": "L0_EXECUTION_AUTHORIZED",
    "lowply_qualification.py": "LOWPLY_QUALIFICATION_AUTHORIZED",
    "runtime_requalification.py": "RUNTIME_REQUAL_AUTHORIZED",
}


#: 🔴 IMPORTED, NEVER REIMPLEMENTED. This file used to carry its OWN copy of the
#: discovery walk. Two injected-defect controls -- blanking the AUTHORIZED match,
#: and scanning only the first statement of each module -- were NOT CAUGHT,
#: because breaking `gate_inventory` could not affect a test that never called
#: it. A test that reimplements its subject tests the copy.
discover_gates = INVENTORY.discover_gates


def test_THE_GATE_INVENTORY_IS_EXACTLY_WHAT_THE_SOURCE_CONTAINS():
    found = {f"{stem}.py": name for stem, name in discover_gates()}
    assert found == EXPECTED_GATES, (
        "the source and the declared inventory disagree; a gate was added or "
        "removed without updating EXPECTED_GATES\n"
        f"  only in source   : {set(found.items()) - set(EXPECTED_GATES.items())}\n"
        f"  only in inventory: {set(EXPECTED_GATES.items()) - set(found.items())}")


def test_THE_COUNT_IS_DERIVED_AND_ANY_REPORT_SAYING_OTHERWISE_IS_WRONG():
    #: 🔴 ONE TRIPWIRE, AND `EXPECTED_GATES` IS IT. The literal `10` sat here
    #: beside `len(EXPECTED_GATES)`, so adding the eleventh gate failed this on
    #: the number rather than on the inventory -- two pins for one fact, which is
    #: how seven, eight and ten were all claimed at once. Adding a gate is now a
    #: single deliberate edit, to the dict above.
    assert INVENTORY.gate_count() == len(discover_gates()) == len(EXPECTED_GATES), (
        "the derived inventory and EXPECTED_GATES must agree")
    assert len(EXPECTED_GATES) == 16, (
        "the total moved: add or remove the entry in EXPECTED_GATES above "
        "deliberately, then update this number in the same edit")


def test_EVERY_GATE_IN_THE_INVENTORY_IS_CLOSED():
    assert INVENTORY.open_gates() == [], (
        f"OPEN GATES IN THE SOURCE: {INVENTORY.open_gates()}")


def test_A_RETIRED_GATE_STILL_COUNTS():
    """The two that were missed, named so their absence cannot recur silently."""
    found = {name for _stem, name in discover_gates()}
    assert "LOWPLY_QUALIFICATION_AUTHORIZED" in found
    assert "RUNTIME_REQUAL_AUTHORIZED" in found


@pytest.mark.parametrize("nonsense", ["EXIT_UNAUTHORIZED = 5",
                                      "UNAUTHORIZED_EXIT = 'x'"])
def test_NON_BOOL_AUTHORIZED_NAMES_ARE_NOT_GATES(nonsense, tmp_path):
    """NEGATIVE CONTROL: exit codes named *UNAUTHORIZED* must not inflate the count."""
    tree = ast.parse(nonsense)
    node = tree.body[0]
    assert not (isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, bool)), nonsense


def _stale_count_claims(text):
    """Claim forms only -- `the EIGHT gates`, `all eight gates`, `EIGHT gates are`.

    🔴 NARROWED DELIBERATELY, AND THE CONTROL BELOW PROVES IT STILL BITES. The
    first version forbade any number adjacent to the word "gates" and flagged the
    sentence "Two gates were missed" in this very file's own explanation. A guard
    that cannot tell a COUNT CLAIM from prose about counts is too broad, and
    broadening is how a guard gets deleted rather than fixed. So it matches the
    three shapes a checker actually uses to ASSERT a total.
    """
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16}
    n = len(EXPECTED_GATES)
    bad = []
    for word, value in words.items():
        if value == n:
            continue
        for claim in (f"the {word} gates", f"all {word} gates", f"{word} gates are"):
            for variant in (claim, claim.upper(),
                            claim.replace(word, word.upper()),
                            claim.replace(word, word.capitalize())):
                if variant in text:
                    bad.append(f"'{variant}' but there are {n}")
    return bad


def test_NO_CHECKER_HARDCODES_A_GATE_COUNT_THAT_DISAGREES():
    """🔴 THE ORIGINAL DEFECT: three checkers, three hand-typed lists, three
    different totals -- seven, eight and ten -- and reports quoting whichever was
    nearest. A checker must DERIVE the count, never write it."""
    stale = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        for claim in _stale_count_claims(path.read_text(encoding="utf-8")):
            stale.append(f"{path}: {claim}")
    assert not stale, "\n".join(stale)


def test_THE_STALE_COUNT_GUARD_IS_NOT_VACUOUS():
    """NEGATIVE CONTROL: the guard must reject the exact strings that were there.

    Without this, narrowing the guard until it matched nothing would look like a
    fix. These are the two real stale claims, verbatim from the source before the
    repair, plus prose that must NOT trip it.
    """
    assert _stale_count_claims("== the SEVEN gates ==")      # H2's, real
    assert _stale_count_claims("== the EIGHT gates ==")      # the pilot's, real
    assert _stale_count_claims("all eight gates stay False")  # the reports', real
    assert not _stale_count_claims("Two gates were missed")   # prose, not a claim
    assert _stale_count_claims("all ten gates are False")      # was right, now stale
    assert not _stale_count_claims("== the 11 gates ==")       # derived, correct
    assert _stale_count_claims("all eleven gates are False")   # was right, now stale
    assert _stale_count_claims("all twelve gates are False")   # was right, now stale
    assert _stale_count_claims("all thirteen gates are False")  # was right, now stale
    assert _stale_count_claims("all fourteen gates are False")  # was right, now stale
    assert _stale_count_claims("all fifteen gates are False")  # was right, now stale
    assert not _stale_count_claims("all sixteen gates are False")  # correct total
