import numpy as np
import random


class FrozenLakeEnv:
    def __init__(self, env_map):
        self.env_map = env_map
        self.n_rows = len(env_map)
        self.n_cols = len(env_map[0])
        self.start_pos = self.find_position("s")
        self.goal_pos = self.find_position("g")
        self.traps = self.find_all_positions("f")
        self.actions = ["left", "right", "up", "down"]
        self.action_map = {
            "left": (0, -1),
            "right": (0, 1),
            "up": (-1, 0),
            "down": (1, 0)
        }
        self.reset()

    def find_position(self, symbol):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.env_map[r][c] == symbol:
                    return (r, c)
        return None

    def find_all_positions(self, symbol):
        positions = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.env_map[r][c] == symbol:
                    positions.append((r, c))
        return positions

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        dr, dc = self.action_map[action]
        new_r = max(0, min(self.n_rows - 1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.n_cols - 1, self.agent_pos[1] + dc))
        self.agent_pos = (new_r, new_c)

        if self.agent_pos in self.traps:
            return self.agent_pos, -1, True  # 掉入陷阱
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 1, True  # 到达目标
        return self.agent_pos, 0, False  # 继续游戏


class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = {
            (r, c): {a: 0 for a in env.actions} for r in range(env.n_rows) for c in range(env.n_cols)
        }

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)  # 探索
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # 利用

    def train(self):
        for _ in range(self.episodes):
            state = self.env.reset()
            action = self.choose_action(state)

            while True:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state) if not done else None

                # SARSA 更新
                if not done:
                    self.q_table[state][action] += self.alpha * (
                            reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action]
                    )
                else:
                    self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])

                if done:
                    break

                state, action = next_state, next_action

    def get_policy(self):
        return {
            (r, c): max(self.q_table[(r, c)], key=self.q_table[(r, c)].get) for r in range(self.env.n_rows) for c in
            range(self.env.n_cols)
        }


if __name__ == "__main__":
    env_map = [
        ["s", "b", "f", "b", "b", "b", "b", "b", "b", "b"],
        ["f", "b", "b", "f", "f", "b", "f", "f", "b", "f"],
        ["b", "b", "f", "b", "f", "b", "b", "f", "b", "f"],
        ["b", "f", "f", "b", "b", "b", "f", "b", "f", "f"],
        ["b", "b", "b", "b", "b", "b", "b", "b", "f", "b"],
        ["b", "b", "f", "f", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "f", "b", "f", "f", "b", "b", "b"],
        ["f", "b", "f", "b", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "f", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "b", "b", "b", "b", "b", "b", "g"]
    ]

    env = FrozenLakeEnv(env_map)
    sarsa = SARSA(env, episodes=1000)
    sarsa.train()
    policy = sarsa.get_policy()

    for r in range(env.n_rows):
        print([policy[(r, c)] for c in range(env.n_cols)])
