from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .format import GameRecord, Move


@dataclass(frozen=True)
class ReplayPaths:
    games_dir: Path  # e.g. scripts/GPU/logs/games

    def game_path(self, config_hash: str, game_id: str) -> Path:
        return self.games_dir / config_hash / f"game-{game_id}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_game_record(paths: ReplayPaths, record: GameRecord) -> Path:
    p = paths.game_path(record.config_hash, record.id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def new_record(*, config_hash: str, depth: int, seed: int, winner: str, moves: List[Move], meta: Optional[Dict[str, Any]] = None) -> GameRecord:
    return GameRecord(
        id=str(uuid.uuid4()),
        timestamp=now_iso(),
        config_hash=config_hash,
        depth=int(depth),
        seed=int(seed),
        winner=winner,
        moves=moves,
        meta=dict(meta or {}),
    )
