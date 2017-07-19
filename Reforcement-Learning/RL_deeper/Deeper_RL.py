import numpy as np
import pandas as pd

N_STATES = 6  # 1维世界的宽度
ACTIONS = ['left', 'right']  # 探索者的可用动作
EPSILON = 0.9  # 贪婪度 greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 移动间隔时间


def init_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros(n_states, len(actions)), columns=actions)
    return table


def choose_action(state, q_table):
    '''采用epsilon-greedy策略选择动作'''
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_reward(state, action):
    if action == 'left':  # move left
        r = 0
        if state == 0:
            s_ = 0
        else:
            s_ = state - 1
    else:  # move right
        r = 0
        s_ = state + 1
        if state == N_STATES - 2:
            r = 1
        elif state == N_STATES - 1:
            s_ = 'terminal'
    return s_, r


def rl():
    q_table = init_q_table(N_STATES, ACTIONS)
    for episode_i in range(MAX_EPISODES):
        s = 0
        is_end = False
        while not is_end:
            a = choose_action(s, q_table)
            s_, r = get_reward(s, a)
            Q = q_table.ix[s, a]
            # TODO 环境的更新
