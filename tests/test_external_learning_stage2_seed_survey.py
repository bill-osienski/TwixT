"""The stage-2 one-seed survey / collision-proof script: pins, record reading, expected state, the two
substituted control sets. Synthetic only -- every in-band value is computed from the band's lower bound,
so no 2026-band literal is written here."""
import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/external-engine-learning-stage2-seed-survey/01_seed_survey.py"


@pytest.fixture(scope="module")
def SV():
    spec = importlib.util.spec_from_file_location("_el2_seed_survey", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def method(SV):
    return SV.load_method()


def _block(S):
    return (S.BAND[0] + 90_000, S.BAND[0] + 90_001)


def _clean_block(S):
    """For the controls that need a CLEAN block under the REAL registries. AMENDED 2026-10-09 at the stage-3
    registration: `_block` (band start + 90,000) is now that registered block. Band start + 95,000 is
    unregistered, outside every floor, and not 10,000-aligned, so the candidate rule can never register it."""
    return (S.BAND[0] + 95_000, S.BAND[0] + 95_000 + 1)

def _record(SV, S, block, *, verdict=None, dirty=False, slots=None):
    lo, hi = block
    lines = [f"HEAD {'a' * 40}; working tree {'DIRTY' if dirty else 'clean'}",
             f"registry {S.REGISTRY_SOURCE} sha256 {'b' * 64}",
             f"CANDIDATE RULE: highest registered/named-spent SEED in the band = {lo - 7}; BASE = {lo}; "
             f"block = [{lo}, {hi}) -- 1 seed",
             f"THE BLOCK ({lo}, {hi})  n=1  direct=NONE  streams=0  injective=True  gap=99 (floor 1)  CLEAN=True",
             *(slots if slots is not None else SV.slot_lines(block)),
             "controls passed: 5/5", f"SURVEY VERDICT   : {verdict or SV.VERDICT_OK}"]
    return "\n".join(lines) + "\n"


def test_the_method_is_the_pinned_tactical_pair_with_this_lines_values(SV, method):
    S, P, S2 = method
    assert (S.N, S.PLAN, S.OWN_RECORDS) == (1, S2.PLAN, SV.OWN_RECORDS) and len(SV.OWN_RECORDS) == 4
    spec = importlib.util.spec_from_file_location(
        "_el1_seed_survey", ROOT / "docs/superpowers/evidence/external-engine-learning-seed-survey/01_seed_survey.py")
    stage1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)
    assert SV.PINNED == stage1.PINNED                                # the same pinned method stage 1 used
    assert callable(P.strict_files) and callable(S.check)


def test_a_changed_method_file_is_refused(SV, monkeypatch):
    rel, _sha = SV.PINNED["survey"]
    monkeypatch.setitem(SV.PINNED, "survey", (rel, "0" * 64))
    with pytest.raises(SV.Refused, match="not the pinned file"):
        SV.load_method()


def test_the_record_reader_accepts_only_the_frozen_rule(SV, method):
    S, _P, _S2 = method
    block = _block(S)
    rec = SV.read_record(_record(SV, S, block), S)
    assert rec["block"] == block and rec["base"] == block[0]
    with pytest.raises(SV.Refused, match="dirty"):
        SV.read_record(_record(SV, S, block, dirty=True), S)
    with pytest.raises(SV.Refused, match="not CLEAN"):
        SV.read_record(_record(SV, S, block, verdict="NOT CLEAN -- STOP"), S)
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, block, slots=SV.slot_lines((block[0] + 1, block[1] + 1))), S)
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, (block[0] + 5, block[1] + 5)), S)        # not a multiple of 10,000
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, (block[0], block[1] + 1)), S)            # two seeds


def test_the_expected_state_holds_only_for_the_right_registry_and_plan(SV, method):
    S, P, S2 = method
    block = _block(S)
    rec = {"head": "a" * 40, "registry_sha256": "b" * 64, "base": block[0], "block": block}
    results = SV._state_controls(S, P, S2, rec, "c" * 64)
    assert len(results) == 11 and all(ok for _label, ok, _detail in results), results


def test_the_SUBSTITUTED_controls_pass_on_a_one_seed_block_and_the_PINNED_ones_would_not(SV, method):
    """Why the substitution: the pinned plants sit at lo+3 .. lo+9 -- outside a one-seed block, where the
    check rightly calls the block clean -- so the pinned controls fail there for the wrong reason."""
    S, _P, _S2 = method
    block = _clean_block(S)
    registries, named, gen, masks, consumed = S.prior_terms()
    prior = (registries, named, gen, consumed, {})
    assert S.check(block, prior, masks)["clean"]
    mine = SV.plant_controls(S, block, prior, masks, {}, True) + SV.exemption_controls(S, block)
    assert len(mine) == 14 and all(ok for _l, ok, _d in mine), mine
    pinned = S._plant_controls(block, prior, masks, {}, True)
    assert [label for label, ok, _d in pinned if not ok] == [
        "a PLAIN candidate seed (.py)", "a candidate seed in a .log -- a suffix H5's filter skipped",
        "a candidate seed inside a .gz", "a candidate seed after an undecodable byte after a digit"]
    assert [label for label, ok, _d in S._exemption_controls(block) if not ok] == [   # its plan holds lo + 1
        "an own record (exact path) is exempt", "the plan holding only block seeds is exempt"]


@pytest.mark.parametrize("indent", [None, 1, 2])
def test_the_serialized_plan_keeps_its_exemption_under_the_unchanged_scanner(SV, method, tmp_path, indent):
    """The plan registration would write -- the seed, a survey hash, the manifest hash -- through the pinned
    repo_literals: its only band literal is the seed, so it is exempt."""
    S, _P, S2 = method
    block = _block(S)
    plan = S2.build_plan(block[0], hashlib.sha256(b"survey").hexdigest())
    path = tmp_path / S2.PLAN
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=indent, sort_keys=True) + "\n")
    assert S.scan(path.read_text()).keys() == {block[0]}
    assert S.repo_literals([path], block, tmp_path) == {}
    path.write_text(json.dumps({**plan, "seed_end": block[1]}, indent=indent) + "\n")
    assert set(S.repo_literals([path], block, tmp_path)) == {block[0], block[1]}
