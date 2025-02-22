### Q-learning Method
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
env = [["s", "b", "f", "b", "b", "b", "b", "b", "b", "b"],
       ["f", "b", "b", "f", "f", "b", "f", "f", "b", "f"],
       ["b", "b", "f", "b", "f", "b", "b", "f", "b", "f"],
       ["b", "f", "f", "b", "b", "b", "f", "b", "f", "f"],
       ["b", "b", "b", "b", "b", "b", "b", "b", "f", "b"],
       ["b", "b", "f", "f", "b", "b", "b", "b", "b", "b"],
       ["b", "b", "b", "f", "b", "f", "f", "b", "b", "b"],
       ["f", "b", "f", "b", "b", "b", "b", "b", "b", "b"],
       ["b", "b", "b", "f", "b", "b", "b", "b", "b", "b"],
       ["b", "b", "b", "b", "f", "b", "b", "b", "b", "g"]]

episode = 1000
V = [0 for _ in range(100)]
V_times = [0 for _ in range(100)]
Q = [0 for _ in range(400)]
Q_times = [0 for _ in range(400)]
gamma = 0.99
epsilon = 0.05
alpha = 0.1
policy_list = []


def reward(i, j):
    # print(env[i][j], i * 4 + j + 1)
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
    index = p.i * 10 + p.j
    lst = Q[index * 4:index * 4 + 4]
    sorted_lst = sorted(lst, reverse=True)
    max_value = max(lst)
    best_actions = [a + 1 for a in range(4) if lst[a] == max_value]  # 找到所有最优动作
    # print(sorted_lst)
    # print(num)
    if num < 1 - epsilon:
        return random.choice(best_actions)
        # return lst.index(sorted_lst[0]) + 1
    else:
        return random.randint(1, 4)


def loop():
    global epsilon
    count = 0
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
                # print(a)
                move(p, a)
            a_list.append(a)
            s_list.append(p.i * 10 + p.j + 1)

            a0 = a_list[-1]
            # a1
            s0 = s_list[-2]
            s1 = s_list[-1]
            Q0 = Q[s0 * 4 + a0 - 5]
            Q1 = max(Q[s1 * 4 - 4:s1 * 4])
            R = 0
            Q[s0 * 4 + a0 - 5] = Q0 + alpha * (R + gamma * Q1 - Q0)
        if reward(p.i, p.j) == 1 and count==0:
            print("Success at episode", t)
            count+=1
        a0 = a_list[-1]
        # a1
        s0 = s_list[-2]
        s1 = s_list[-1]
        Q0 = Q[s0 * 4 + a0 - 5]
        Q1 = max(Q[s1 * 4 - 4:s1 * 4])
        R = reward(p.i, p.j)
        Q[s0 * 4 + a0 - 5] = Q0 + alpha * (R + gamma * Q1 - Q0)
        # print(Q)
        # print(t)
        # epsilon = max(0.1, epsilon * 0.9999)
        # print(epsilon)


def move(p, direction):
    if direction == 1 and p.i != 0:
        p.i += -1
    elif direction == 2 and p.i != 9:
        p.i += 1
    elif direction == 3 and p.j != 0:
        p.j += -1
    elif direction == 4 and p.j != 9:
        p.j += 1


def policy():
    for i in range(100):
        lst = Q[i * 4:i * 4 + 4]
        max_index = lst.index(max(lst)) + 1
        policy_list.append(max_index)
    policy_matrix=np.array(policy_list).reshape(10, 10)
    print(policy_matrix)


def examine():
    p = Person()
    step = 0
    while reward(p.i, p.j) == 0 and step < 100:
        move(p, policy_list[p.i * 10 + p.j])
        step += 1
    if reward(p.i, p.j) == 1:
        print("Success!")
    elif reward(p.i, p.j) == -1:
        print("Fail!")
    else:
        print("Endless!")

# 策略动作映射
action_arrows = {1: "↑", 2: "↓", 3: "←", 4: "→"}

# 计算每个状态各个方向的 Q 值
def get_q_values():
    Q_values = np.zeros((10, 10, 4))  # 4x4网格, 每个网格有4个方向的Q值
    for i in range(10):
        for j in range(10):
            state_index = i * 10 + j
            Q_values[i, j, :] = Q[state_index * 4:state_index * 4 + 4]  # 获取4个方向的Q值
    return Q_values

# 画策略+Q值热力图（风车样式展示四个方向的Q值）
def plot_grid_policy():
    Q_values = get_q_values()  # 获取Q值矩阵
    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.cm.RdYlGn  # 热力颜色映射

    # 在每个格子上绘制策略箭头 & 特殊标记
    for i in range(10):
        for j in range(10):
            state_index = i * 10 + j

            # 特殊状态的固定颜色
            if env[i][j] == "f":  # 陷阱 (灰色)
                rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1, color='gray')
                ax.add_patch(rect)
                ax.text(j, i, "×", ha="center", va="center", fontsize=16, fontweight="bold", color="black")

            elif env[i][j] == "g":  # 终点 (金色)
                rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1, color='gold')
                ax.add_patch(rect)
                ax.text(j, i, "GOAL", ha="center", va="center", fontsize=12, fontweight="bold", color="black")

            else:
                # 获取当前状态四个方向的Q值
                q_up, q_down, q_left, q_right = Q_values[i, j]
                norm = plt.Normalize(vmin=np.min(Q_values), vmax=np.max(Q_values))

                # 画上方小三角形 (↑)
                triangle_up = Polygon([[j, i], [j - 0.5, i - 0.5], [j + 0.5, i - 0.5]], 
                                      color=cmap(norm(q_up)))
                ax.add_patch(triangle_up)
                ax.text(j, i - 0.3, f"{q_up:.2f}", ha="center", va="center", fontsize=8, color="black")

                # 画下方小三角形 (↓)
                triangle_down = Polygon([[j, i], [j - 0.5, i + 0.5], [j + 0.5, i + 0.5]], 
                                        color=cmap(norm(q_down)))
                ax.add_patch(triangle_down)
                ax.text(j, i + 0.3, f"{q_down:.2f}", ha="center", va="center", fontsize=8, color="black")

                # 画左方小三角形 (←)
                triangle_left = Polygon([[j, i], [j - 0.5, i - 0.5], [j - 0.5, i + 0.5]], 
                                        color=cmap(norm(q_left)))
                ax.add_patch(triangle_left)
                ax.text(j - 0.3, i, f"{q_left:.2f}", ha="center", va="center", fontsize=8, color="black")

                # 画右方小三角形 (→)
                triangle_right = Polygon([[j, i], [j + 0.5, i - 0.5], [j + 0.5, i + 0.5]], 
                                         color=cmap(norm(q_right)))
                ax.add_patch(triangle_right)
                ax.text(j + 0.3, i, f"{q_right:.2f}", ha="center", va="center", fontsize=8, color="black")

                # 显示最优策略箭头
                best_action = policy_list[state_index]
                ax.text(j, i, action_arrows[best_action], ha="center", va="center", fontsize=16, fontweight="bold", color="black")

    # 设置图像格式
    ax.set_xticks(np.arange(-0.5, 10, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")  # 隐藏坐标轴
    plt.grid(color="black", linestyle="-", linewidth=2)

    plt.title("Gridworld Policy with Full Directional Q-values Heatmap", fontsize=16)
    plt.gca().invert_yaxis()  # 使坐标原点在左下角
    plt.show()


def main():
    loop()
    policy()
    examine()
    plot_grid_policy()

if __name__ == "__main__":
    main()
