from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .engine import SimOutcome


@dataclass
class GameSummary:
    games: int
    red: int
    black: int
    draws: int

    @property
    def bias(self) -> float:
        denom = max(1, self.games - self.draws)
        return (self.red - self.black) / denom


def summarize(outcomes: List[SimOutcome]) -> GameSummary:
    red = sum(1 for o in outcomes if o.winner == "red")
    black = sum(1 for o in outcomes if o.winner == "black")
    draws = sum(1 for o in outcomes if o.winner == "draw")
    return GameSummary(games=len(outcomes), red=red, black=black, draws=draws)
