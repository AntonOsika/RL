from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from domain import Episode


@dataclass
class Config:
    n_episodes: int
    max_episode_length: int
    n_actions: int
    n_inp_dim: int
    n_hidden_dim: int
    batch_size: int
    gamma: float


class ValueModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.l1 = torch.nn.Linear(config.n_inp_dim, config.n_hidden_dim)
        self.l2 = torch.nn.Linear(config.n_hidden_dim, config.n_hidden_dim)
        self.l3 = torch.nn.Linear(config.n_hidden_dim, config.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.l3(
            F.leaky_relu(
                self.l2(
                    F.leaky_relu(
                        self.l1(
                            x
                        )
                    )
                )
            )
        )

    def step(self, xs, actions, ys):
        qs = self(torch.FloatTensor(xs))
        vs = qs[torch.LongTensor(range(len(actions))), torch.LongTensor(actions)]
        loss = F.mse_loss(vs, torch.FloatTensor(ys))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q(self, state):
        return self(torch.FloatTensor(state)).detach().numpy()


class Agent:
    def __init__(self, config: Config):
        self.q_model = ValueModel(config)

    def q(self, state):
        return self.q_model.get_q(state)

    def act(self, state):
        qs = self.q(state)
        exp = np.exp(qs)

        # Softmax sample:
        return np.random.choice(len(qs), p=exp / exp.sum())

    def train(self, batch: List[Episode]):
        xs = [step.state for episode in batch for step in episode.raw_episode]
        actions = [step.action for episode in batch for step in episode.raw_episode]
        ys = [q for episode in batch for q in episode.qs]

        self.q_model.step(xs, actions, ys)
