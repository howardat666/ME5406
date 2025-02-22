import numpy as np
import random


# 自定义环境类
class FrozenLake:
    def __init__(self, custom_map):
        self.map = np.array(custom_map)
        self.grid_size = len(custom_map)
        self.start = (0, 0)
        self.goal = (self.grid_size - 1, self.grid_size - 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._pos_to_state(self.state)

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # 上、下、左、右
        x, y = self.state

        dx, dy = moves[action]
        new_x = max(0, min(x + dx, self.grid_size - 1))
        new_y = max(0, min(y + dy, self.grid_size - 1))
        self.state = (new_x, new_y)

        cell = self.map[new_x][new_y]
        done = (cell == 'g') or (cell == 'f')
        reward = 1 if cell == 'g' else (-1 if cell == 'f' else 0)

        return self._pos_to_state(self.state), reward, done

    def _pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]


# 蒙特卡洛控制实现
def monte_carlo_control(env, episodes=10000, gamma=0.99, epsilon=0.1):
    n_states = env.grid_size ** 2
    n_actions = 4

    # 初始化Q表和行为策略
    Q = np.zeros((n_states, n_actions))
    policy = np.ones((n_states, n_actions)) / n_actions  # 初始随机策略

    # 保存首次访问的回报
    returns = {}

    for i in range(episodes):
        # 生成轨迹
        trajectory = []
        state = env.reset()
        done = False

        # 采集完整轨迹
        while not done:
            action = np.random.choice(n_actions, p=policy[state])
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        # 计算首次访问的回报
        G = 0
        visited = set()

        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = gamma * G + reward

            # 仅更新首次访问
            if (state, action) not in visited:
                visited.add((state, action))
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)

                # 更新Q值
                Q[state][action] = np.mean(returns[(state, action)])

        # 更新ε-greedy策略
        for s in range(n_states):
            best_action = np.argmax(Q[s])
            policy[s] = epsilon / n_actions
            policy[s][best_action] += 1 - epsilon
        print(i)
    return Q, policy


# 使用示例
if __name__ == "__main__":
    custom_map = [
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

    env = FrozenLake(custom_map)
    Q, policy = monte_carlo_control(env, episodes=10000)

    # 展示最优策略
    optimal_policy = np.argmax(policy, axis=1)
    print("最优策略矩阵：")
    print(optimal_policy.reshape(10, 10))

    # 示例轨迹测试
    state = env.reset()
    done = False
    print("\n示例路径：")
    step=0
    while not done and step<50:
        action = optimal_policy[state]
        next_state, reward, done = env.step(action)
        print(f"State: {state} -> Action: {['上', '下', '左', '右'][action]} -> Reward: {reward}")
        state = next_state
        step+=1