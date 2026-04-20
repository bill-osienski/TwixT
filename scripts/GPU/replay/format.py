from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Move:
    turn: int
    player: str  # "red"|"black"
    row: int
    col: int
    # Optional richer telemetry, populated when the Python engine is real:
    bridges_created: List[Dict[str, Any]] = field(default_factory=list)
    heuristics: Dict[str, float] = field(default_factory=dict)
    search_score: Optional[float] = None


@dataclass
class GameRecord:
    id: str
    timestamp: str
    config_hash: str
    depth: int
    seed: int
    winner: str  # "red"|"black"|"draw"
    moves: List[Move]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["moves"] = [asdict(m) for m in self.moves]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GameRecord":
        moves = [Move(**m) for m in d.get("moves", [])]
        return GameRecord(
            id=str(d.get("id")),
            timestamp=str(d.get("timestamp")),
            config_hash=str(d.get("config_hash")),
            depth=int(d.get("depth")),
            seed=int(d.get("seed")),
            winner=str(d.get("winner")),
            moves=moves,
            meta=dict(d.get("meta", {})),
        )
