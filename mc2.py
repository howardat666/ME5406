import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon, env):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


def print_policy(policy, name):
    actions = ['←', '↓', '→', '↑']
    print(f"\n{name} Policy:")
    for i in range(4):
        print(" ".join(actions[a] for a in policy[i * 4: (i + 1) * 4]))


def monte_carlo():
    env = gym.make("FrozenLake-v1")  # ✅ 使用 Gym 0.21.0 支持的 FrozenLake-v1
    Q_mc = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = 0.9
    epsilon_min = 0.1
    epsilon_decay = 0.999
    alpha = 0.2
    gamma = 0.9
    episodes = 5000
    returns = {}

    for episode in range(episodes):
        state = env.reset()  # ✅ Gym 0.21.0 只返回 state
        episode_data = []
        done = False

        while not done:
            action = epsilon_greedy(Q_mc, state, epsilon, env)
            next_state, reward, done, _ = env.step(action)  # ✅ 只解包 4 个值
            episode_data.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)
                Q_mc[state, action] = np.mean(returns[(state, action)])
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("Monte Carlo Q-table:")
    print(Q_mc)
    policy_mc = np.argmax(Q_mc, axis=1)
    print_policy(policy_mc, "Monte Carlo")


if __name__ == "__main__":
    monte_carlo()
