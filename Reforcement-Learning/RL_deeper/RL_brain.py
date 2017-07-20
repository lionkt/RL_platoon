# 模仿Morvan的例子写的RL的超类，凡是用pandas存储q-table的都可以继承这个类

import numpy as np
import pandas as pd


class RL_basic(object):
    def __init__(self, actions, learning_rate, gamma, epsilon_greedy):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_greedy = epsilon_greedy
        self.q_table = pd.DataFrame(columns=actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() > self.epsilon_greedy:
            action_name = np.random.choice(self.actions)
        else:
            # 为了应对’动作不同，Q值相同’的情况，这里引入了随机打乱index的操作
            state_actions = self.q_table.ix[observation, :]
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            action_name = state_actions.argmax()
        return action_name

    # 由于q-table需要动态的扩展，所以在exploration前检查该state是否存在. 如果不存在要添上
    def check_state_exist(self, observation):
        if observation not in self.q_table.index:
            self.append = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=observation))

    # 由于sarsa和q-learning不一样，所以此处不详细写
    def learn(self, *args):
        pass
