### SARSA 方法 (sarsa.py)
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


def sarsa():
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    Q_sarsa = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    alpha = 0.2
    gamma = 0.95
    episodes = 5000

    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q_sarsa, state, epsilon, env)
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q_sarsa, next_state, epsilon, env)
            Q_sarsa[state, action] += alpha * (
                        reward + gamma * Q_sarsa[next_state, next_action] - Q_sarsa[state, action])
            state, action = next_state, next_action
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("SARSA Q-table:")
    print(Q_sarsa)
    policy_sarsa = np.argmax(Q_sarsa, axis=1)
    print_policy(policy_sarsa, "SARSA")


if __name__ == "__main__":
    sarsa()