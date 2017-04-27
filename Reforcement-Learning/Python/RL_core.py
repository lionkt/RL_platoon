import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RL(object):
    def __init__(self, action_space_I, learning_rate=0.01, reward_decay=0.9):
        self.action_space = action_space_I
        self.lr = learning_rate
        self.gamma = reward_decay

        self.q_table = pd.DataFrame(columns=action_space_I)

    def set_epsilon_greedy_para(self, e_greedy_para_I):
        self.e_greedy_para = e_greedy_para_I
