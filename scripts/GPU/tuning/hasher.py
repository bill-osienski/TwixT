from __future__ import annotations

import hashlib
from typing import Dict

from ..config.knobs import HASH_FIELDS, normalize_to_allowed


def config_hash(knobs: Dict[str, float]) -> str:
    """Compute stable SHA1 from ordered knob fields.

    This is the key to mapping sweep entries <-> validations.
    """
    k = normalize_to_allowed(knobs)
    parts = []
    for name in HASH_FIELDS:
        # Fixed formatting so float string variations don't change hashes
        parts.append(f"{name}={k.get(name, 0.0):.10g}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]
