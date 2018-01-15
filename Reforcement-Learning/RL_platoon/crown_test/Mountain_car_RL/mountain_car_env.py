import random

import math
import numpy as np

#### state space
POS_RANGE = np.array([-1.2, 0.5])
VEL_RANGE = np.array([-0.07, 0.07])
GOAL = 0.5  # mountain-car中flag所在的位置

#### action space
NUM_ACT = 2
ACT = np.array([-1, 1])  # -1-push left, 1-push right


def random_reset(method=None):
    """
    reset state for simulation.
    """
    rnd1 = random.uniform(0, 1)
    rnd2 = random.uniform(0, 1)
    if method == 1:
        pos1 = -0.8
        pos2 = 0.0
        vel1 = -0.02
        vel2 = 0.02
    elif method == 2:   # 接近gym的组合
        pos1 = -0.6
        pos2 = -0.4
        vel1 = -0.02
        vel2 = 0.02
    elif method == 3:   # RBN中的组合
        pos1 = POS_RANGE[0]
        pos2 = POS_RANGE[1]
        vel1 = VEL_RANGE[0]
        vel2 = VEL_RANGE[1]

    # x = np.array(
    #     [((-0.6) - (-0.4)) * rnd1 + (-0.4), 0])  # set according to https://github.com/openai/gym/wiki/MountainCar-v0
    x = np.array([(pos2 - pos1) * rnd1 + pos1, (vel2 - vel1) * rnd2 + vel1])
    observation = x
    return observation


def cal_reward(obs):
    reward = np.abs(obs[0] - (-0.5))  # 距离开始的地方越远，奖励越多
    # reward = 0 if obs[0] >= GOAL else -1
    return reward


def step_next(obs_old, a_old):
    """
    mountain car dynamics environment
    """
    x = np.array([0.0, 0.0], dtype=float)
    temp = obs_old[1] + 0.001 * a_old - 0.0025 * math.cos(3 * obs_old[0])
    x[1] = max(VEL_RANGE[0], min(temp, VEL_RANGE[1]))  # update velocity
    temp = obs_old[0] + x[1]
    x[0] = max(POS_RANGE[0], min(temp, POS_RANGE[1]))  # update position

    if x[0] <= POS_RANGE[0]:
        x[1] = 0
    if x[0] >= GOAL:
        x[0] = GOAL
        x[1] = 0
    done = bool(x[0] >= GOAL)
    observation_n = x
    return observation_n, done
