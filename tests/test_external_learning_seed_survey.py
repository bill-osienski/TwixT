"""The stage-1 seed survey / collision-proof script: pins, record reading, expected state. Synthetic only --
every in-band value is computed from the band's lower bound, so no 2026-band literal is written here."""
import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/external-engine-learning-seed-survey/01_seed_survey.py"


@pytest.fixture(scope="module")
def SV():
    spec = importlib.util.spec_from_file_location("_el_seed_survey", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def method(SV):
    return SV.load_method()


def _record(SV, S, EL, block, *, verdict=None, dirty=False, slots=None):
    lo, hi = block
    lines = [f"HEAD {'a' * 40}; working tree {'DIRTY' if dirty else 'clean'}",
             f"registry {S.REGISTRY_SOURCE} sha256 {'b' * 64}",
             f"CANDIDATE RULE: highest registered/named-spent SEED in the band = {lo - 7}; BASE = {lo}; "
             f"block = [{lo}, {hi}) -- 18 seeds",
             f"THE BLOCK ({lo}, {hi})  n=18  direct=NONE  streams=0  injective=True  gap=99 (floor 18)  CLEAN=True",
             *(slots if slots is not None else SV.slot_lines(EL, block)),
             "controls passed: 5/5", f"SURVEY VERDICT   : {verdict or SV.VERDICT_OK}"]
    return "\n".join(lines) + "\n"


def test_the_method_is_the_pinned_tactical_pair_with_this_lines_values(SV, method):
    S, P, EL = method
    assert (S.N, S.PLAN, S.OWN_RECORDS) == (18, EL.PLAN, SV.OWN_RECORDS)
    assert S.OWN_RECORDS[3] == SV.OUT["post"] and len(SV.OWN_RECORDS) == 4
    assert callable(P.strict_files) and callable(S.check)


def test_a_changed_method_file_is_refused(SV, monkeypatch):
    rel, _sha = SV.PINNED["survey"]
    monkeypatch.setitem(SV.PINNED, "survey", (rel, "0" * 64))
    with pytest.raises(SV.Refused, match="not the pinned file"):
        SV.load_method()


def test_the_record_reader_accepts_only_the_frozen_rule(SV, method):
    S, _P, EL = method
    block = (S.BAND[0] + 90_000, S.BAND[0] + 90_018)
    rec = SV.read_record(_record(SV, S, EL, block), S, EL)
    assert rec["block"] == block and rec["base"] == block[0]
    with pytest.raises(SV.Refused, match="dirty"):
        SV.read_record(_record(SV, S, EL, block, dirty=True), S, EL)
    with pytest.raises(SV.Refused, match="not CLEAN"):
        SV.read_record(_record(SV, S, EL, block, verdict="NOT CLEAN -- STOP"), S, EL)
    moved = SV.slot_lines(EL, block)
    moved[7] = moved[7].replace(str(block[0] + 7), str(block[0] + 17))
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, EL, block, slots=moved), S, EL)
    off = (block[0] + 5, block[1] + 5)                               # not a multiple of 10,000
    with pytest.raises(SV.Refused, match="frozen rule"):
        SV.read_record(_record(SV, S, EL, off), S, EL)


def test_the_expected_state_holds_only_for_the_right_registry_and_plan(SV, method):
    S, P, EL = method
    block = (S.BAND[0] + 90_000, S.BAND[0] + 90_018)
    rec = {"head": "a" * 40, "registry_sha256": "b" * 64, "base": block[0], "block": block}
    results = SV._state_controls(S, P, EL, rec, "c" * 64)
    assert len(results) == 10 and all(ok for _label, ok, _detail in results), results


def test_the_slots_are_the_modules_layout(SV, method):
    S, _P, EL = method
    block = (S.BAND[0] + 90_000, S.BAND[0] + 90_018)
    lines = SV.slot_lines(EL, block)
    assert len(lines) == 18 and len({ln.rsplit(" ", 1)[1] for ln in lines}) == 18


@pytest.mark.parametrize("indent", [None, 2])
def test_the_serialized_plan_keeps_its_exemption_under_the_unchanged_scanner(SV, method, tmp_path, indent):
    """The plan registration would write -- the surveyed block, the record's hash -- through the pinned
    repo_literals: every band literal in it is a slot, so it is exempt. The old encoding, with the
    exclusive endpoint, is the control: the scanner counts all 19 values."""
    S, _P, EL = method
    raw = (ROOT / SV.OUT["survey"]).read_bytes()
    block = SV.read_record(raw.decode("utf-8"), S, EL)["block"]
    plan = EL.build_plan(block, hashlib.sha256(raw).hexdigest())
    path = tmp_path / EL.PLAN
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=indent) + "\n")
    assert S.scan(path.read_text()).keys() == set(range(*block))
    assert S.repo_literals([path], block, tmp_path) == {}
    path.write_text(json.dumps({**plan, "block": list(block)}, indent=indent) + "\n")
    old = S.repo_literals([path], block, tmp_path)
    assert len(old) == 19 and block[1] in old
