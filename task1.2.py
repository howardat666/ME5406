### SARSA Method
import random

env = [["s", "b", "b", "b"],
       ["b", "f", "b", "f"],
       ["b", "b", "b", "f"],
       ["f", "b", "b", "g"]]
episode = 100
V = [0 for _ in range(16)]
V_times = [0 for _ in range(16)]
Q = [0 for _ in range(64)]
Q_times = [0 for _ in range(64)]
gamma = 0.9
epsilon = 0.1
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
    global epsilon, alpha
    count=0
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
            # count+=1
            # alpha = 1 / (count + 1)
            a_list.append(a)
            s_list.append(p.i * 4 + p.j + 1)

            if len(a_list) > 1:
                a0 = a_list[-2]
                a1 = a_list[-1]
                s0 = s_list[-3]
                s1 = s_list[-2]
                Q0 = Q[s0 * 4 + a0 - 5]
                Q1 = Q[s1 * 4 + a1 - 5]
                R = 0
                Q[s0 * 4 + a0 - 5] = Q0 + alpha * (R + gamma * Q1 - Q0)
            # print(p.i, p.j)
        a0 = a_list[-1]
        s0 = s_list[-2]
        s1 = s_list[-1]
        Q0 = Q[s0 * 4 + a0 - 5]
        Q1 = 0
        i = (s1 - 1) // 4
        j = (s1 - 1) % 4
        R = reward(i, j)
        Q[s0 * 4 + a0 - 5] = Q0 + alpha * (R + gamma * Q1 - Q0)

        # alpha = 1 / (t + 1)

        # print(p.i, p.j)
        """
        for k in range(len(s_list) - 1):
            a = a_list[k]
            s = s_list[k]
            Q_0 = Q[s * 4 + a - 5]

            if k == len(s_list) - 2:
                s_final = s_list[-1]
                Q_prime = 0
                i = (s_final - 1) // 4
                j = (s_final - 1) % 4
                R = reward(i, j)
            else:
                a_prime = a_list[k+1]
                s_prime = s_list[k+1]
                Q_prime = Q[s_prime * 4 + a_prime - 5]
                R = 0

            Q[s * 4 + a - 5] = Q_0 + alpha * (R + gamma * Q_prime - Q_0)
            Q_times[s * 4 + a - 5] += 1
        """
        """
        print(a_list)
        g_list = [0 for _ in range(len(s_list))]
        for k in range(len(s_list)):
            if k == 0:
                g_list[len(s_list) - k - 1] = 0
            else:
                g_list[len(s_list) - k - 1] = reward(p.i, p.j) * gamma ** k
        # print(g_list)
        for k in range(len(s_list)):
            s = s_list[k]
            V[s - 1] = (V[s - 1] * V_times[s - 1] + g_list[k]) / (V_times[s - 1] + 1)
            V_times[s - 1] += 1
        """
        # print(V)
        # print(V_times)

        print(Q)
        # print(Q_times)

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
