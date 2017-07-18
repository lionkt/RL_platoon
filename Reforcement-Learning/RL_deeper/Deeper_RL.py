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

def choose_action(s):



def rl():
    q_table = init_q_table(N_STATES, ACTIONS)
    for episode_i in range(MAX_EPISODES):
        s=0

