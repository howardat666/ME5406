import numpy as np
import gym

# 创建 FrozenLake 环境
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# 设置超参数
epsilon = 0.1  # ε-贪心策略中的探索概率
alpha = 0.2  # 学习率
gamma = 0.95  # 折扣因子
episodes = 10000  # 训练回合数

# 初始化 Q 表
Q_mc = np.zeros((env.observation_space.n, env.action_space.n))  # Monte Carlo
Q_sarsa = np.zeros((env.observation_space.n, env.action_space.n))  # SARSA
Q_qlearning = np.zeros((env.observation_space.n, env.action_space.n))  # Q-Learning


def epsilon_greedy(Q, state, epsilon):
    """ ε-贪心策略选择动作 """
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


# --------------------------- Monte Carlo 方法 -------------------------------- #
returns = {}  # 存储每个 (state, action) 对应的回报

for episode in range(episodes):
    state = env.reset()[0]
    episode_data = []
    done = False

    # 生成一条完整的轨迹
    while not done:
        action = epsilon_greedy(Q_mc, state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state

    G = 0  # 初始化回报
    visited = set()
    for state, action, reward in reversed(episode_data):
        G = gamma * G + reward  # 计算折扣回报
        if (state, action) not in visited:  # 只更新首次访问
            visited.add((state, action))
            if (state, action) not in returns:
                returns[(state, action)] = []
            returns[(state, action)].append(G)
            Q_mc[state, action] = np.mean(returns[(state, action)])  # 计算平均回报

# --------------------------- SARSA 方法 -------------------------------- #
for episode in range(episodes):
    state = env.reset()[0]
    action = epsilon_greedy(Q_sarsa, state, epsilon)
    done = False

    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy(Q_sarsa, next_state, epsilon)

        # SARSA 更新公式
        Q_sarsa[state, action] += alpha * (reward + gamma * Q_sarsa[next_state, next_action] - Q_sarsa[state, action])

        state, action = next_state, next_action  # 更新状态和动作

# --------------------------- Q-Learning 方法 -------------------------------- #
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = epsilon_greedy(Q_qlearning, state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Q-Learning 更新公式
        Q_qlearning[state, action] += alpha * (
                    reward + gamma * np.max(Q_qlearning[next_state]) - Q_qlearning[state, action])

        state = next_state  # 更新状态

# 训练结束，打印最终 Q 表
print("Monte Carlo Q-table:")
print(Q_mc)
print("\nSARSA Q-table:")
print(Q_sarsa)
print("\nQ-Learning Q-table:")
print(Q_qlearning)


def extract_policy(Q):
    """从 Q-table 提取最优策略"""
    return np.argmax(Q, axis=1)  # 每个状态选择 Q 值最大的动作

# 计算三个方法的最优策略
policy_mc = extract_policy(Q_mc)
policy_sarsa = extract_policy(Q_sarsa)
policy_qlearning = extract_policy(Q_qlearning)

# 打印策略
def print_policy(policy, name):
    """打印策略，输出对应的动作"""
    actions = ['←', '↓', '→', '↑']  # 对应 OpenAI Gym FrozenLake 动作：0-Left, 1-Down, 2-Right, 3-Up
    print(f"\n{name} Policy:")
    for i in range(4):  # 假设是 4x4 冰湖
        print(" ".join(actions[a] for a in policy[i * 4: (i + 1) * 4]))

# 输出策略
print_policy(policy_mc, "Monte Carlo")
print_policy(policy_sarsa, "SARSA")
print_policy(policy_qlearning, "Q-Learning")

