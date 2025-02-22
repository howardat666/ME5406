### Q-Learning 方法 (q_learning.py)
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


def q_learning():
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    Q_qlearning = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    alpha = 0.2
    gamma = 0.95
    episodes = 5000

    for episode in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            action = epsilon_greedy(Q_qlearning, state, epsilon, env)
            next_state, reward, done, _, _ = env.step(action)
            Q_qlearning[state, action] += alpha * (
                        reward + gamma * np.max(Q_qlearning[next_state]) - Q_qlearning[state, action])
            state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("Q-Learning Q-table:")
    print(Q_qlearning)
    policy_qlearning = np.argmax(Q_qlearning, axis=1)
    print_policy(policy_qlearning, "Q-Learning")


if __name__ == "__main__":
    q_learning()
