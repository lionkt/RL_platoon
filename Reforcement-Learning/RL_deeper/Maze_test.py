import numpy as np
import pandas as pd

import RL_brain as RL
from Maze import Maze

MAX_EPISODE = 100


# 从RL_basic类继承的Q-learning算法
class QLearning(RL.RL_basic):
    def __init__(self, actions, learning_rate, gamma, epsilon_greedy):
        super(QLearning, self).__init__(actions=actions, learning_rate=learning_rate, gamma=gamma,
                                        epsilon_greedy=epsilon_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)  # 先保证s_在q_table里面
        q_predict = self.q_table.ix[s, a]
        if s_ == 'terminal':
            q_target = r  # 没有下一步了，只能单纯的返回reward了
        else:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        self.q_table.ix[s, a] += self.learning_rate * (q_target - q_predict)


# 从RL_basic类继承的Sarsa-learning算法
class SarsaLearning(RL.RL_basic):
    def __init__(self, actions, learning_rate, gamma, epsilon_greedy):
        super(SarsaLearning, self).__init__(actions=actions, learning_rate=learning_rate, gamma=gamma,
                                            epsilon_greedy=epsilon_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ == 'terminal':
            q_target = r  # 没有下一步了，只能单纯的返回reward了
        else:
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        self.q_table.ix[s, a] += self.learning_rate * (q_target - q_predict)


def update():
    for episode_i in range(MAX_EPISODE):
        s = env.reset()
        while True:
            env.render()  # 刷新环境的显示
            a = RL_function.choose_action(str(s))
            s_, r, done = env.step(a)
            RL_function.learn(s=str(s), a=a, r=r, s_=str(s_))
            s = s_
            if done:
                break
    print('reach MAX_EPISODE')
    env.destroy()  # 注意有意识销毁环境，释放内存


if __name__ == "__main__":
    env = Maze()
    gamma = 0.9
    lr = 0.01
    eg = 0.9
    RL_function = QLearning(actions=list(range(env.n_actions)), learning_rate=lr, gamma=gamma, epsilon_greedy=eg)

    env.after(100, update())  # tkinter的写法
    env.mainloop()
