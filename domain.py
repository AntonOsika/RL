from dataclasses import dataclass
from typing import Any, List


@dataclass
class Step:
    state: Any
    action: int
    reward: float
    terminal: bool

@dataclass
class Episode:
    raw_episode: List[Step]
    qs: List[float]



