import pytest
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    _parse_args, manifest_is_neutral, resolve_integrity_csv, resolve_fpu_values,
    resolve_output_paths, PROTOCOL_FPUS, DEFAULT_A_MANIFEST, DEFAULT_PHASE0_CSV,
    DEFAULT_OUT, DEFAULT_FPUS)


def test_cli_aliases_and_none_sentinels():
    assert _parse_args(["--manifest", "m"]).manifest == "m"
    assert _parse_args(["--a-manifest", "m"]).manifest == "m"
    assert _parse_args(["--integrity-csv", "i"]).integrity_csv == "i"
    assert _parse_args(["--phase0-csv", "i"]).integrity_csv == "i"
    a = _parse_args([])
    assert a.integrity_csv is None and a.fpu_values is None
    assert a.out is None and a.summary_out is None and a.strata_summary_out is None
    assert a.skip_integrity_check is False and a.allow_non_protocol_fpu is False


def test_manifest_is_neutral_is_strict():
    assert manifest_is_neutral([{"ply_bucket": "midgame"}, {"ply_bucket": "late"}]) is True
    assert manifest_is_neutral([{"case_id": "c"}, {"case_id": "d"}]) is False
    with pytest.raises(ValueError):
        manifest_is_neutral([{"ply_bucket": "late"}, {"case_id": "d"}])   # mixed
    with pytest.raises(ValueError):
        manifest_is_neutral([])                                           # empty


def test_resolve_integrity_conditional():
    assert resolve_integrity_csv(None, False, False, DEFAULT_PHASE0_CSV) == DEFAULT_PHASE0_CSV
    assert resolve_integrity_csv(None, False, True, DEFAULT_PHASE0_CSV) is None
    assert resolve_integrity_csv("x", False, True, DEFAULT_PHASE0_CSV) == "x"
    assert resolve_integrity_csv(None, True, False, DEFAULT_PHASE0_CSV) is None


def test_resolve_fpu_values_frozen_protocol():
    assert resolve_fpu_values(None, True, False) == PROTOCOL_FPUS       # neutral default
    assert resolve_fpu_values(None, False, False) == [float(x) for x in DEFAULT_FPUS.split(",")]
    assert resolve_fpu_values("0.0,-0.20", True, False) == [0.0, -0.20]
    with pytest.raises(SystemExit):
        resolve_fpu_values("0.0,-0.10,-0.20", True, False)             # non-protocol, no override
    assert resolve_fpu_values("0.0,-0.10,-0.20", True, True) == [0.0, -0.10, -0.20]
    with pytest.raises(SystemExit):
        resolve_fpu_values("-0.20", True, True)                        # missing baseline 0.0


def test_resolve_output_paths_mode_scoped():
    lo = resolve_output_paths(None, None, None, "a/x.csv", False)
    assert lo[0] == DEFAULT_OUT                                         # legacy -> A defaults
    ne = resolve_output_paths(None, None, None, "logs/eval/v16a/m.csv", True)
    assert ne == ("logs/eval/v16a/neutral_fpu_sweep_cases.csv",
                  "logs/eval/v16a/neutral_fpu_sweep_summary.csv",
                  "logs/eval/v16a/neutral_fpu_sweep_by_stratum.csv")
    assert resolve_output_paths("o", "s", "t", "m.csv", True) == ("o", "s", "t")  # explicit wins
