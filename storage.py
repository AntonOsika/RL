from typing import Any, List, Tuple

import numpy as np

from agent import Config
from domain import Episode, Step


class Memory:
    def __init__(self, config: Config):
        self.episodes = []
        self.config = config

    def store(self, raw_episode: List[Step], final_v: float):
        qs = []
        q = final_v
        for step in reversed(raw_episode):
            q *= self.config.gamma
            q += step.reward
            qs.append(q)
        self.episodes.append(
            Episode(raw_episode, list(reversed(qs)))
        )

    def size(self):
        return len(self.episodes)

    def sample_batch(self, size: int) -> List[Episode]:
        idxs = np.random.randint(range(self.size()), size, replace=False)
        return [self.episodes[idx] for idx in idxs]

