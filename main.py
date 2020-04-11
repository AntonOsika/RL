from typing import List

import gym

from agent import Config, Agent
from storage import Memory
from storage import Step


def main():
    config = Config(
        n_episodes=10000,
        max_episode_length=200,
        n_actions=2,
        n_inp_dim=4,
        n_hidden_dim=64,
        batch_size=1,
        gamma=0.99,
    )

    env = gym.make('CartPole-v0').unwrapped
    memory = Memory(config)
    agent = Agent(config)

    for _ in range(config.n_episodes):
        episode: List[Step] = []
        s = env.reset()
        final_v = 0
        for _ in range(config.max_episode_length):
            a = agent.act(s)
            s2, r, t, _ = env.step(a)
            episode.append(Step(state=s, action=a, reward=r, terminal=t))
            s = s2
            if t:
                break
        else:
            # If no break
            final_v = agent.q(s).max()
        memory.store(episode, final_v)

        print(f"Reward: {sum(step.reward for step in episode)}")

        # Always train on last episode:
        agent.train(memory.episodes[-1:])


if __name__ == '__main__':
    main()
