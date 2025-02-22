import random

# 4x4 环境地图
env = [["s", "b", "b", "b"],
       ["b", "f", "b", "f"],
       ["b", "b", "b", "f"],
       ["f", "b", "b", "g"]]

# 训练参数
episode = 10000  # 训练轮数
gamma = 0.9  # 折扣因子
epsilon = 0.5  # 初始探索率
alpha = 0.1  # 学习率

# 状态值 V(s)
V = [0 for _ in range(16)]
V_times = [0 for _ in range(16)]

# 状态-动作值 Q(s, a)
Q = [0 for _ in range(64)]
Q_times = [0 for _ in range(64)]

# 计算奖励函数
def reward(i, j):
    if env[i][j] == "f":  # 失败格子
        return -1
    elif env[i][j] == "g":  # 目标格子
        return 1
    else:  # 普通格子
        return 0

# 代理（智能体）
class Person:
    def __init__(self):
        self.i = 0
        self.j = 0

# 选择动作的策略（ε-贪心策略）
def greedy(p):
    num = random.random()
    index = p.i * 4 + p.j
    lst = [Q[index * 4 + a] for a in range(4)]  # 取该状态的 4 个 Q 值
    max_value = max(lst)
    best_actions = [a + 1 for a in range(4) if lst[a] == max_value]  # 找到所有最优动作

    if num < 1 - epsilon:
        return random.choice(best_actions)  # 选择最优动作
    else:
        return random.randint(1, 4)  # 以 epsilon 选择随机动作

# 执行行动
def move(p, direction):
    new_i, new_j = p.i, p.j
    if direction == 1 and new_i > 0:  # 上
        new_i -= 1
    elif direction == 2 and new_i < 3:  # 下
        new_i += 1
    elif direction == 3 and new_j > 0:  # 左
        new_j -= 1
    elif direction == 4 and new_j < 3:  # 右
        new_j += 1
    p.i, p.j = new_i, new_j  # 更新坐标

# 训练循环
def loop():
    global epsilon
    for t in range(episode):
        p = Person()
        s_list = [p.i * 4 + p.j + 1]  # 记录状态序列
        a_list = []  # 记录动作序列
        r_list = []  # 记录奖励序列

        while env[p.i][p.j] not in ["f", "g"]:  # 直到终点（失败或成功）
            a = greedy(p)  # 选择动作
            move(p, a)  # 执行动作
            a_list.append(a)
            s_list.append(p.i * 4 + p.j + 1)
            r_list.append(reward(p.i, p.j))

        # 计算折扣回报 G
        G = 0
        for k in range(len(r_list) - 1, -1, -1):
            G = r_list[k] + gamma * G  # 计算 G 值
            s = s_list[k]
            V[s - 1] = (1 - alpha) * V[s - 1] + alpha * G  # 增量更新 V(s)

            if k < len(a_list):
                a = a_list[k]
                Q[s * 4 + a - 5] = (1 - alpha) * Q[s * 4 + a - 5] + alpha * G  # 增量更新 Q(s, a)

        # 逐渐减少探索率 epsilon（防止过度随机探索）
        epsilon = max(0.01, epsilon * 0.99)

# 计算最终策略
def policy():
    policy_list = []
    for i in range(16):
        lst = Q[i * 4:i * 4 + 4]  # 取该状态的 4 个动作的 Q 值
        max_value = max(lst)
        best_actions = [a for a in range(4) if lst[a] == max_value]
        policy_list.append(random.choice(best_actions) + 1)  # 随机选择最优动作
    print("最优策略：", policy_list)

# 主函数
def main():
    loop()
    policy()

if __name__ == "__main__":
    main()
