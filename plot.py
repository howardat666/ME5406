import random
import numpy as np
import matplotlib.pyplot as plt

env = [["s", "b", "b", "b"],
       ["b", "f", "b", "f"],
       ["b", "b", "b", "f"],
       ["f", "b", "b", "g"]]
episode = 5000
gamma = 1
epsilon = 0.2
policy_list = []
Q = [0 for _ in range(64)]
Q_times = [0 for _ in range(64)]


def reward(i, j):
    if env[i][j] == "f":
        return -1
    elif env[i][j] == "g":
        return 1
    else:
        return 0


class Person:
    def __init__(self):
        self.i = 0
        self.j = 0


def greedy(p):
    num = random.random()
    index = p.i * 4 + p.j
    lst = Q[index * 4:index * 4 + 4]
    max_value = max(lst)
    best_actions = [a + 1 for a in range(4) if lst[a] == max_value]  # 找到所有最优动作
    if num < 1 - epsilon:
        return random.choice(best_actions)
    else:
        return random.randint(1, 4)


def loop():
    global epsilon
    for t in range(episode):
        p = Person()
        s_list = [1]
        a_list = []
        while reward(p.i, p.j) == 0:
            if t == 0:
                a = random.randint(1, 4)
                move(p, a)
            else:
                a = greedy(p)
                move(p, a)
            a_list.append(a)
            s_list.append(p.i * 4 + p.j + 1)
        g_list = [0 for _ in range(len(s_list))]
        for k in range(len(s_list)):
            if k == 0:
                g_list[len(s_list) - k - 1] = 0
            else:
                g_list[len(s_list) - k - 1] = reward(p.i, p.j) * gamma ** (k - 1)

        for k in range(len(s_list) - 1):
            a = a_list[k]
            s = s_list[k]
            Q[s * 4 + a - 5] = (Q[s * 4 + a - 5] * Q_times[s * 4 + a - 5] + g_list[k]) / (Q_times[s * 4 + a - 5] + 1)
            Q_times[s * 4 + a - 5] += 1


def move(p, direction):
    if direction == 1 and p.i != 0:
        p.i += -1
    elif direction == 2 and p.i != 3:
        p.i += 1
    elif direction == 3 and p.j != 0:
        p.j += -1
    elif direction == 4 and p.j != 3:
        p.j += 1


def policy():
    for i in range(16):
        lst = Q[i * 4:i * 4 + 4]
        max_index = lst.index(max(lst)) + 1
        policy_list.append(max_index)


def examine():
    p = Person()
    step = 0
    while reward(p.i, p.j) == 0 and step < 100:
        move(p, policy_list[p.i * 4 + p.j])
        step += 1
    if reward(p.i, p.j) == 1:
        print("Success!")
    elif reward(p.i, p.j) == -1:
        print("Fail!")
    else:
        print("Endless!")


# 策略动作映射
action_arrows = {1: "↑", 2: "↓", 3: "←", 4: "→"}

# 计算最大 Q 值的热力颜色
def get_q_value_matrix():
    Q_values = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            state_index = i * 4 + j
            Q_values[i, j] = max(Q[state_index * 4:state_index * 4 + 4])  # 取最大 Q 值
    return Q_values

# 画策略+Q值热力图
def plot_grid_policy():
    Q_values = get_q_value_matrix()  # 获取最大 Q 值矩阵
    fig, ax = plt.subplots(figsize=(6, 6))

    # 画网格背景（热力图）
    cax = ax.imshow(Q_values, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(cax, label="Q value")

    # 在每个格子上绘制策略箭头 & 特殊标记
    for i in range(4):
        for j in range(4):
            state_index = i * 4 + j
            # if env[i][j] == "s":  # 起点
            #     ax.text(j, i, "START", ha="center", va="center", fontsize=12, fontweight="bold", color="black")
            if env[i][j] == "g":  # 终点
                ax.text(j, i, "GOAL", ha="center", va="center", fontsize=12, fontweight="bold", color="black")
            elif env[i][j] == "f":  # 陷阱
                ax.text(j, i, "×", ha="center", va="center", fontsize=16, fontweight="bold", color="black")
            else:
                # 显示策略箭头
                best_action = policy_list[state_index]
                ax.text(j, i, action_arrows[best_action], ha="center", va="center", fontsize=16, fontweight="bold", color="black")
                # 显示 Q 值数值
                ax.text(j, i + 0.3, f"{Q_values[i, j]:.2f}", ha="center", va="center", fontsize=10, color="black")

    # 设定图像格式
    ax.set_xticks(np.arange(-0.5, 4, 1))
    ax.set_yticks(np.arange(-0.5, 4, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")  # 完全隐藏坐标轴
    plt.title("Gridworld Policy with Q-values Heatmap")
    plt.show()


def main():
    loop()
    policy()
    examine()
    plot_grid_policy()

if __name__ == "__main__":
    main()
