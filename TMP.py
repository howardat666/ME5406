
import gym
from gym import spaces
import numpy as np
import random

class MazeEnv(gym.Env):
    """
    自定义迷宫环境，继承自 gym.Env
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEnv, self).__init__()
        # 定义动作空间和状态空间
        # 动作空间：上、下、左、右
        self.action_space = spaces.Discrete(4)
        # 状态空间：智能体在迷宫中的位置（二维坐标）
        self.maze_size = (5, 5)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)

        # 定义迷宫（0 表示空地，-1 表示墙壁）
        self.maze = np.zeros(self.maze_size)
        self.maze[0, 3] = -1  # 墙壁位置
        self.maze[1, 1] = -1
        self.maze[1, 3] = -1
        self.maze[2, 1] = -1
        self.maze[3, 3] = -1
        self.maze[4, 1] = -1

        # 起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (0, 4)

        # 智能体初始位置
        self.agent_pos = self.start_pos

    def step(self, action):
        """
        执行动作
        """
        # 定义动作对应的移动
        directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        # 根据动作计算新的位置
        move = directions[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # 默认的奖励和终止标志
        reward = -1
        done = False

        # 检查新位置是否在迷宫范围内
        if (0 <= new_pos[0] < self.maze_size[0]) and (0 <= new_pos[1] < self.maze_size[1]):
            # 检查新位置是否是墙壁
            if self.maze[new_pos] == -1:
                # 撞到墙壁
                reward = -5
            else:
                # 合法移动
                self.agent_pos = new_pos
        else:
            # 超出迷宫范围
            reward = -5

        # 检查是否到达终点
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True

        obs = np.array(self.agent_pos)
        info = {}
        return obs, reward, done, info

    def reset(self):
        """
        重置环境到初始状态
        """
        self.agent_pos = self.start_pos
        return np.array(self.agent_pos)

    def render(self, mode='human'):
        """
        渲染迷宫环境
        """
        maze_render = np.copy(self.maze)
        maze_render[self.agent_pos] = 2  # 智能体的位置
        maze_render[self.start_pos] = 3  # 起点
        maze_render[self.goal_pos] = 4   # 终点
        symbol_map = {
            -1: 'W',  # 墙壁
            0: ' ',   # 空地
            2: 'A',   # 智能体
            3: 'S',   # 起点
            4: 'G'    # 终点
        }
        print("\n".join(["".join([symbol_map[item] for item in row]) for row in maze_render]))
        print("\n")

def mc_control_on_policy(env, num_episodes=5000, gamma=1.0, epsilon=0.1):
    """
    基于第一访问蒙特卡洛的 on-policy 控制（ε-贪心）。
    :param env: 自定义迷宫环境
    :param num_episodes: 训练的回合数
    :param gamma: 折扣因子
    :param epsilon: 探索率
    :return: Q, 最优的状态-动作价值函数
    """
    # Q 表示状态-动作价值函数，大小为 [行, 列, 动作数]
    Q = np.zeros((env.maze_size[0], env.maze_size[1], env.action_space.n))

    # 这里使用一个字典来存储每个状态-动作对的回报（列表），方便后续取平均做更新
    returns = dict()
    for r in range(env.maze_size[0]):
        for c in range(env.maze_size[1]):
            for a in range(env.action_space.n):
                returns[((r, c), a)] = []

    def epsilon_greedy_policy(state):
        """
        给定当前的 Q 和 explored state, 采用 ε-贪心策略选择动作
        """
        r, c = state
        if random.random() < epsilon:
            # 随机探索
            return np.random.choice(env.action_space.n)
        else:
            # 贪心选择
            return np.argmax(Q[r, c])

    for episode in range(num_episodes):
        # 生成一条回合（episode）
        state = env.reset()
        episode_trace = []  # 存储 (state, action, reward) 元组

        done = False
        while not done:
            action = epsilon_greedy_policy(tuple(state))
            next_state, reward, done, _ = env.step(action)
            episode_trace.append((tuple(state), action, reward))
            state = next_state

        # 回溯回合，更新 Q
        visited_state_actions = set()
        G = 0  # 从后往前计算折扣回报
        # 在这里从后向前计算更简洁（若想从前向后可先沿 episode_trace 再次扫一遍计算回报）
        for t in reversed(range(len(episode_trace))):
            s_t, a_t, r_t = episode_trace[t]
            G = gamma * G + r_t
            # 检查是否是该回合中首次出现的 (s_t, a_t)
            if (s_t, a_t) not in visited_state_actions:
                visited_state_actions.add((s_t, a_t))
                returns[(s_t, a_t)].append(G)
                # 增量方式更新 Q(s, a)
                Q[s_t[0], s_t[1], a_t] = np.mean(returns[(s_t, a_t)])
    return Q

if __name__ == "__main__":
    # 创建环境
    env = MazeEnv()

    # 使用蒙特卡洛方法进行训练
    Q = mc_control_on_policy(env, num_episodes=3000, gamma=1.0, epsilon=0.1)

    # 打印最终学到的 Q
    print("训练结束后学到的状态-动作价值函数 Q：")
    for r in range(env.maze_size[0]):
        for c in range(env.maze_size[1]):
            print(f"State=({r},{c}) -> Q={Q[r, c]}")
        print()

    # 根据学到的 Q 构造出一个贪心策略并测试
    def greedy_policy(state):
        return np.argmax(Q[state[0], state[1]])

    # 测试智能体在环境中的表现
    state = env.reset()
    env.render()
    done = False
    step_count = 0
    while not done and step_count < 50:  # 做一个简单的步数限制，防止卡死
        action = greedy_policy(tuple(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        step_count += 1

    if tuple(state) == env.goal_pos:
        print("智能体成功到达目标！")
    else:
        print("智能体未能到达目标。")