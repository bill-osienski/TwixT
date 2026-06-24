"""Deterministic builder for the Targeted Value Calibration v2 mixed manifest.

Correction rows (hard target) + retention rows (anchored to a checkpoint's own
probe_black_root_value) are merged into one CSV the calibration pool can load.
See docs/superpowers/specs/2026-06-23-targeted-value-calibration-v2-design.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def resolve_anchor_rows(rows: list, anchor_label: str) -> list:
    """Rows whose checkpoint == anchor_label (exact); else the unique set whose
    checkpoint endswith ':' + anchor_label. Raise on ambiguous or missing."""
    exact = [r for r in rows if r.get("checkpoint") == anchor_label]
    if exact:
        return exact
    suffix = [r for r in rows if str(r.get("checkpoint", "")).endswith(":" + anchor_label)]
    labels = sorted({r["checkpoint"] for r in suffix})
    if len(labels) == 1:
        return suffix
    if len(labels) > 1:
        raise ValueError(
            f"ambiguous anchor label {anchor_label!r}; candidates: {labels}; "
            f"pass an exact --*-anchor-label")
    raise ValueError(f"no checkpoint matches anchor label {anchor_label!r}")
