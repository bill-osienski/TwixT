from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from .knobs import HASH_FIELDS, KNOB_SPECS, normalize_to_allowed


@dataclass(frozen=True)
class SearchConfigIO:
    """Read/write helper for `assets/js/ai/search.json`.

    - Preserves unknown keys on write.
    - Extracts known numeric knobs (including any new knobs you add to KNOB_SPECS).
    """

    path: Path

    def load(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        knobs: Dict[str, float] = {}

        # Prefer: explicit knob names we know about
        for name in KNOB_SPECS.keys():
            if name in raw and isinstance(raw[name], (int, float)):
                knobs[name] = float(raw[name])

        # Ensure hash fields exist (default missing)
        for name in HASH_FIELDS:
            if name not in knobs:
                knobs[name] = float(KNOB_SPECS[name].default)

        knobs = normalize_to_allowed(knobs)
        return knobs, raw

    def save(self, knobs: Dict[str, float], raw: Dict[str, Any]) -> None:
        out = dict(raw)
        for k, v in knobs.items():
            # only write knobs we recognize; leave all else intact
            if k in KNOB_SPECS:
                out[k] = float(v)
        self.path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_search_json(path: Path) -> Dict[str, float]:
    knobs, _ = SearchConfigIO(path).load()
    return knobs


def write_search_json(path: Path, knobs: Dict[str, float]) -> None:
    io = SearchConfigIO(path)
    _, raw = io.load() if path.exists() else ({}, {})
    io.save(knobs, raw)
