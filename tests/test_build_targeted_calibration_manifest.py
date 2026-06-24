import pytest

from scripts.GPU.alphazero.build_targeted_calibration_manifest import resolve_anchor_rows


def _rows(*labels):
    return [{"checkpoint": l, "case_id": f"c{i}"} for i, l in enumerate(labels)]


def test_resolve_exact_match():
    out = resolve_anchor_rows(_rows("0001", "0379", "0001"), "0001")
    assert [r["checkpoint"] for r in out] == ["0001", "0001"]


def test_resolve_unique_suffix_match():
    out = resolve_anchor_rows(_rows("alphazero-v2-calib020-from0409:0001", "x:0379"), "0001")
    assert [r["checkpoint"] for r in out] == ["alphazero-v2-calib020-from0409:0001"]


def test_resolve_ambiguous_suffix_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_anchor_rows(_rows("a:0001", "b:0001"), "0001")


def test_resolve_no_match_raises():
    with pytest.raises(ValueError, match="no checkpoint matches"):
        resolve_anchor_rows(_rows("0379", "0409"), "0001")
