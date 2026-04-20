from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def norm_status(s: str) -> str:
    return (s or "UNTESTED").strip().upper()


@dataclass
class CandidateEntry:
    status: str = "UNTESTED"
    sweep_runs: int = 0
    validation_runs: int = 0
    streak60: int = 0
    last_score: Optional[float] = None
    last_bias: Optional[float] = None
    retired_reason: Optional[str] = None
    per_depth_bias: Optional[Dict[int, float]] = None  # Per-depth bias tracking


@dataclass
class TuningState:
    iteration: int = 0
    active_hash: Optional[str] = None

    # Hash -> entry
    hash_registry: Dict[str, CandidateEntry] = field(default_factory=dict)

    # Training samples for predicted-bias/correlation models
    samples: List[Dict[str, Any]] = field(default_factory=list)

    # Bookkeeping
    best_score: float = 999.0
    cycles_since_improvement: int = 0

    def get(self, h: str) -> CandidateEntry:
        if h not in self.hash_registry:
            self.hash_registry[h] = CandidateEntry()
        return self.hash_registry[h]

    def mark_status(self, h: str, status: str) -> None:
        e = self.get(h)
        e.status = norm_status(status)


def load_state(path: Path) -> TuningState:
    if not path.exists():
        return TuningState()
    raw = json.loads(path.read_text(encoding="utf-8"))

    st = TuningState(
        iteration=int(raw.get("iteration", 0)),
        active_hash=raw.get("active_hash"),
        best_score=float(raw.get("best_score", 999.0)),
        cycles_since_improvement=int(raw.get("cycles_since_improvement", 0)),
    )
    st.samples = list(raw.get("samples", []))

    reg = raw.get("hash_registry", {}) or {}
    for h, ent in reg.items():
        # Convert per_depth_bias keys back to int
        pdb = ent.get("per_depth_bias")
        if pdb:
            pdb = {int(k): float(v) for k, v in pdb.items()}
        st.hash_registry[h] = CandidateEntry(
            status=ent.get("status", "UNTESTED"),
            sweep_runs=int(ent.get("sweep_runs", 0)),
            validation_runs=int(ent.get("validation_runs", 0)),
            streak60=int(ent.get("streak60", 0)),
            last_score=ent.get("last_score"),
            last_bias=ent.get("last_bias"),
            retired_reason=ent.get("retired_reason"),
            per_depth_bias=pdb,
        )

    return st


def save_state(path: Path, st: TuningState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {
        "iteration": st.iteration,
        "active_hash": st.active_hash,
        "best_score": st.best_score,
        "cycles_since_improvement": st.cycles_since_improvement,
        "samples": st.samples,
        "hash_registry": {
            h: {
                "status": e.status,
                "sweep_runs": e.sweep_runs,
                "validation_runs": e.validation_runs,
                "streak60": e.streak60,
                "last_score": e.last_score,
                "last_bias": e.last_bias,
                "retired_reason": e.retired_reason,
                "per_depth_bias": e.per_depth_bias,
            }
            for h, e in st.hash_registry.items()
        },
    }
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
