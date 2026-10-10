"""The stage-3 game-seed survey / collision-proof script: pins, record reading, expected state, the pinned
controls on a 32-seed block. Synthetic only -- every in-band value is computed from the band's lower bound,
so no 2026-band literal is written here."""
import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/external-engine-learning-stage3-seed-survey/01_seed_survey.py"


@pytest.fixture(scope="module")
def SV():
    spec = importlib.util.spec_from_file_location("_el3_seed_survey", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def method(SV):
    return SV.load_method()


def _block(S):
    return (S.BAND[0] + 90_000, S.BAND[0] + 90_032)


def _clean_block(S):
    """For the controls that need a CLEAN block under the REAL registries. AMENDED 2026-10-09 at the stage-3
    registration: `_block` (band start + 90,000) is now that registered block. Band start + 95,000 is
    unregistered, outside every floor, and not 10,000-aligned, so the candidate rule can never register it."""
    return (S.BAND[0] + 95_000, S.BAND[0] + 95_000 + 32)

def _record(SV, S, S3, block, *, verdict=None, dirty=False, slots=None):
    lo, hi = block
    lines = [f"HEAD {'a' * 40}; working tree {'DIRTY' if dirty else 'clean'}",
             f"registry {S.REGISTRY_SOURCE} sha256 {'b' * 64}",
             f"CANDIDATE RULE: highest registered/named-spent SEED in the band = {lo - 7}; BASE = {lo}; "
             f"block = [{lo}, {hi}) -- 32 seeds",
             f"THE BLOCK ({lo}, {hi})  n=32  direct=NONE  streams=0  injective=True  gap=99 (floor 32)  CLEAN=True",
             *(slots if slots is not None else SV.slot_lines(S3, block)),
             "controls passed: 5/5", f"SURVEY VERDICT   : {verdict or SV.VERDICT_OK}"]
    return "\n".join(lines) + "\n"


def test_the_method_is_the_pinned_tactical_pair_with_this_lines_values(SV, method):
    S, P, S3 = method
    assert (S.N, S.PLAN, S.OWN_RECORDS) == (32, S3.PLAN, SV.OWN_RECORDS) and len(SV.OWN_RECORDS) == 4
    spec = importlib.util.spec_from_file_location(
        "_el1_seed_survey_for3", ROOT / "docs/superpowers/evidence/external-engine-learning-seed-survey/01_seed_survey.py")
    stage1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)
    assert SV.PINNED == stage1.PINNED                                # the same pinned method stages 1 and 2 used
    assert callable(P.strict_files) and callable(S.check)


def test_a_changed_method_file_is_refused(SV, monkeypatch):
    rel, _sha = SV.PINNED["survey"]
    monkeypatch.setitem(SV.PINNED, "survey", (rel, "0" * 64))
    with pytest.raises(SV.Refused, match="not the pinned file"):
        SV.load_method()


def test_the_record_reader_accepts_only_the_frozen_rule(SV, method):
    S, _P, S3 = method
    block = _block(S)
    rec = SV.read_record(_record(SV, S, S3, block), S, S3)
    assert rec["block"] == block and rec["base"] == block[0]
    with pytest.raises(SV.Refused, match="dirty"):
        SV.read_record(_record(SV, S, S3, block, dirty=True), S, S3)
    with pytest.raises(SV.Refused, match="not CLEAN"):
        SV.read_record(_record(SV, S, S3, block, verdict="NOT CLEAN -- STOP"), S, S3)
    moved = SV.slot_lines(S3, block)
    moved[7] = moved[7].replace(str(block[0] + 7), str(block[0] + 17))
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, S3, block, slots=moved), S, S3)
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, S3, (block[0] + 5, block[1] + 5)), S, S3)   # not a multiple of 10,000
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, S3, (block[0], block[1] - 14)), S, S3)      # 18 seeds, stage 1's size


def test_the_expected_state_holds_only_for_the_right_registry_and_plan(SV, method):
    S, P, S3 = method
    block = _block(S)
    rec = {"head": "a" * 40, "registry_sha256": "b" * 64, "base": block[0], "block": block}
    results = SV._state_controls(S, P, S3, rec, "c" * 64)
    assert len(results) == 11 and all(ok for _label, ok, _detail in results), results


def test_the_slots_are_the_modules_layout_one_seed_per_game(SV, method):
    S, _P, S3 = method
    block = _block(S)
    lines = SV.slot_lines(S3, block)
    assert len(lines) == 32 and [int(ln.rsplit(" ", 1)[1]) for ln in lines] == list(range(*block))
    assert [ln.split()[6] for ln in lines] == ["red", "black"] * 16


def test_the_PINNED_plant_and_exemption_controls_hold_on_a_32_seed_block__none_substituted(SV, method):
    S, _P, _S3 = method
    block = _clean_block(S)
    registries, named, gen, masks, consumed = S.prior_terms()
    prior = (registries, named, gen, consumed, {})
    assert S.check(block, prior, masks)["clean"]
    results = S._plant_controls(block, prior, masks, {}, True) + S._exemption_controls(block)
    assert results and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


@pytest.mark.parametrize("indent", [None, 1, 2])
def test_the_serialized_plan_keeps_its_exemption_under_the_unchanged_scanner(SV, method, tmp_path, indent):
    """The plan registration would write -- base, slot count, the 32-slot layout, the survey and manifest
    hashes -- through the pinned repo_literals: every band literal in it is a slot, so it is exempt. With
    the exclusive endpoint added, the control: the scanner counts all 33 values."""
    S, _P, S3 = method
    block = _block(S)
    plan = S3.build_plan(block[0], hashlib.sha256(b"survey").hexdigest())
    path = tmp_path / S3.PLAN
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=indent, sort_keys=True) + "\n")
    assert S.scan(path.read_text()).keys() == set(range(*block))
    assert S.repo_literals([path], block, tmp_path) == {}
    path.write_text(json.dumps({**plan, "block_end": block[1]}, indent=indent) + "\n")
    old = S.repo_literals([path], block, tmp_path)
    assert len(old) == 33 and block[1] in old
