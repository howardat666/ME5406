### Monte Carlo Method
import random

env = [["s", "b", "b", "b"],
       ["b", "f", "b", "f"],
       ["b", "b", "b", "f"],
       ["f", "b", "b", "g"]]
episode = 5000
V = [0 for _ in range(16)]
V_times = [0 for _ in range(16)]
Q = [0 for _ in range(64)]
Q_times = [0 for _ in range(64)]
gamma = 1
epsilon = 0.2
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
    index = p.i * 4 + p.j
    lst = Q[index * 4:index * 4 + 4]
    sorted_lst = sorted(lst, reverse=True)
    # max_value = max(lst)
    # best_actions = [a + 1 for a in range(4) if lst[a] == max_value]  # 找到所有最优动作
    # print(sorted_lst)
    # print(num)
    if num < 1 - epsilon:
        # return random.choice(best_actions)
        return lst.index(sorted_lst[0]) + 1
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
                # print(a)
                move(p, a)
            a_list.append(a)
            s_list.append(p.i * 4 + p.j + 1)
            # print(p.i, p.j)
        # print(p.i, p.j)
        # print(a_list)
        g_list = [0 for _ in range(len(s_list))]
        for k in range(len(s_list)):
            if k == 0:
                g_list[len(s_list) - k - 1] = 0
            else:
                g_list[len(s_list) - k - 1] = reward(p.i, p.j) * gamma ** (k - 1)
        # print(g_list)
        for k in range(len(s_list)):
            s = s_list[k]
            V[s - 1] = (V[s - 1] * V_times[s - 1] + g_list[k]) / (V_times[s - 1] + 1)
            V_times[s - 1] += 1
        # print(V)
        # print(V_times)

        for k in range(len(s_list) - 1):
            a = a_list[k]
            s = s_list[k]
            Q[s * 4 + a - 5] = (Q[s * 4 + a - 5] * Q_times[s * 4 + a - 5] + g_list[k]) / (Q_times[s * 4 + a - 5] + 1)
            Q_times[s * 4 + a - 5] += 1
        print(Q)
        print(Q_times)

        # epsilon = max(0.01, epsilon * 0.99)


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
    print(policy_list)


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


def main():
    loop()
    policy()
    examine()


if __name__ == "__main__":
    main()
