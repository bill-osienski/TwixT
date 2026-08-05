"""Atlas artifact schema, provenance validation and emission.

Every undefined value stays None through emission. A missing boundary is
FLAGGED, never defaulted -- a zero-filled record is indistinguishable from a
real one.

The Stage 3 producer document is stored ONCE, undivided, under `snapshots`, so
an artifact row is directly consumable by Read-outs A, B and C with no
translation layer between them to drift. The row holds NATIVE Python -- tuple
keys, LegResult and BoundaryRecord dataclasses -- and `_jsonable` normalizes it
at the JSON boundary, exactly where Stage 2 put that concern.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .build_atlas_corpus import _jsonable

# 1, not 2: no artifact of this schema has ever been emitted, and a version
# number implying a predecessor invites a reader to hunt for one.
ROW_SCHEMA_VERSION = 1


def build_row(*, game_idx: int, replay_seed: int, target_ply: int, phase: str,
              side: str, split: str, inherited_I: int, reset_count: int,
              reset_rate: Optional[float], last_reset_ply: Optional[int],
              boundary: Optional[Any], legs: Sequence[Any],
              label: str, features_at_boundary: Optional[Dict[str, Any]],
              features_at_400: Optional[Dict[str, Any]],
              snapshots: Dict[str, Any], flat_policy: bool,
              near_even: bool) -> Dict[str, Any]:
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "game_idx": game_idx, "replay_seed": replay_seed,
        "target_ply": target_ply, "phase": phase, "side": side, "split": split,
        "inherited_I": inherited_I,
        # Section 2b: reset statistics are explicit, and every row is kept.
        "reset_count": reset_count, "reset_rate": reset_rate,
        "last_reset_ply": last_reset_ply,
        "boundary": boundary, "boundary_missing": boundary is None,
        # LegResult objects, NOT vars()-flattened dicts: Read-out B and
        # atlas_labelling read `l.nominal_B` by ATTRIBUTE, so a flattened row
        # could not be handed to calibrate_gate at all. `_jsonable` converts
        # them at emission.
        "legs": legs, "label": label,
        # BOTH captures: B=400 supplies section 6's 400-tree diagnostic
        # contrast. Together with `label` this row IS a Read-out A row.
        "features_at_boundary": features_at_boundary,
        "features_at_400": features_at_400,
        # The Stage 3 document, WHOLE and under the key Read-out C consumes:
        # tracer snapshots, captures, both parent-visit maps and both deep
        # lines. Splitting it into overlapping copies is how they drift, and
        # storing it under any other name forces a surrogate row in between.
        "snapshots": snapshots,
        # Strata facts, so Read-outs B and C need no second source.
        "flat_policy": flat_policy, "near_even": near_even,
    }


_SHA1 = re.compile(r"[0-9a-fA-F]{40}\Z")


def validate_provenance(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fails CLOSED. A dirty tree or unidentifiable checkpoint means the run is
    not reconstructible, whatever its numbers say.

    Digests must be HEXADECIMAL, not merely 40 characters: a placeholder, a
    truncated path or a typo'd ref can be 40 characters long and is not a SHA-1.
    """
    prov = prov or {}
    problems = []
    if prov.get("worktree_clean") is not True:
        problems.append("worktree_clean")
    for field in ("checkpoint_sha1", "git_head"):
        value = prov.get(field)
        if not isinstance(value, str) or not _SHA1.match(value):
            problems.append(field)
    return {"verdict": "PROVENANCE_FAILURE" if problems else "OK",
            "problems": problems}


def emit(run: Dict[str, Any]) -> str:
    """Serialize through _jsonable, but ONLY for a run that validates.

    The provenance gate lives here because emission is the one point every
    artifact passes through. A fail-closed check that nothing calls is
    decoration.

    Validation runs BEFORE serialization so a payload defect still raises
    TypeError rather than being masked by the gate. NO default=str -- it would
    stringify a schema defect into a plausible-looking value instead of failing.
    """
    checked = validate_provenance(run.get("provenance"))
    if checked["verdict"] != "OK":
        raise ValueError(
            f"refusing to emit: provenance does not validate "
            f"({', '.join(checked['problems'])})")
    return json.dumps(_jsonable(run), indent=2, sort_keys=True)
