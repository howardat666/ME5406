import numpy as np
import random

# 设定地图
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

# 定义动作空间
actions = ["left", "right", "up", "down"]
action_map = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0)
}

# 查找特殊位置
def find_position(env, symbol):
    for r in range(len(env)):
        for c in range(len(env[0])):
            if env[r][c] == symbol:
                return (r, c)
    return None

def find_all_positions(env, symbol):
    return [(r, c) for r in range(len(env)) for c in range(len(env[0])) if env[r][c] == symbol]

# 环境初始化
start_pos = find_position(env_map, "s")
goal_pos = find_position(env_map, "g")
traps = find_all_positions(env_map, "f")

# 代理移动
def step(state, action):
    r, c = state
    dr, dc = action_map[action]
    new_r = max(0, min(len(env_map) - 1, r + dr))
    new_c = max(0, min(len(env_map[0]) - 1, c + dc))
    new_state = (new_r, new_c)

    if new_state in traps:
        return new_state, -1, True  # 掉入陷阱
    if new_state == goal_pos:
        return new_state, 1, True  # 到达终点
    return new_state, 0, False  # 正常移动

# 选择动作（epsilon-greedy策略）
def choose_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # 探索
    return max(q_table[state], key=q_table[state].get)  # 利用

# SARSA训练
def train_sarsa(env_map, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = {(r, c): {a: 0 for a in actions} for r in range(len(env_map)) for c in range(len(env_map[0]))}

    for _ in range(episodes):
        state = start_pos
        action = choose_action(q_table, state, epsilon)

        while True:
            next_state, reward, done = step(state, action)
            next_action = choose_action(q_table, next_state, epsilon) if not done else None

            # SARSA 更新
            if not done:
                q_table[state][action] += alpha * (
                    reward + gamma * q_table[next_state][next_action] - q_table[state][action]
                )
            else:
                q_table[state][action] += alpha * (reward - q_table[state][action])

            if done:
                break

            state, action = next_state, next_action

    return q_table

# 获取最优策略
def get_policy(q_table):
    return {(r, c): max(q_table[(r, c)], key=q_table[(r, c)].get) for r in range(len(env_map)) for c in range(len(env_map[0]))}

# 运行SARSA算法
q_table = train_sarsa(env_map, episodes=1000)
policy = get_policy(q_table)

# 输出最优策略
for r in range(len(env_map)):
    print([policy[(r, c)] for c in range(len(env_map[0]))])
