import math
import random
import numpy as np
import learning_param

#### state space
POS_RANGE = np.mat([[-1.2], [0.6]])
VEL_RANGE = np.mat([[-0.07], [0.07]])
GOAL = 0.5  # mountain-car中flag所在的位置
POS_MAP_RANGE = np.mat([[0], [1]])
VEL_MAP_RANGE = np.mat([[0], [1]])
GRID_SIZE = np.mat([[4], [4]])
NUM_STATE_FEATURES = GRID_SIZE[1] * GRID_SIZE[2]

c_map_pos = np.mat([[POS_RANGE[1], 1], [POS_RANGE[2], 1]])
c_map_pos = c_map_pos.I * POS_RANGE
c_map_vel = np.mat([VEL_RANGE[1], 1], [VEL_RANGE[2], 1])
c_map_vel = c_map_vel.I * VEL_RANGE
GRID_STEP = np.mat([(POS_MAP_RANGE[2] - POS_MAP_RANGE[1]) / GRID_SIZE[1]],
                   [(VEL_MAP_RANGE[2] - VEL_MAP_RANGE[1]) / GRID_SIZE[2]])
## fill in the center of RBF
GRID_CENTERS = np.mat(np.zeros(2, NUM_STATE_FEATURES))
for i in range(GRID_SIZE[1]):
    for j in range(GRID_SIZE[2]):
        GRID_CENTERS[:, ((i - 1) * GRID_SIZE[2]) + j] = np.mat([[POS_MAP_RANGE[1] + ((i - 0.5) * GRID_STEP[1])],
                                                                [VEL_MAP_RANGE[1] + ((j - 0.5) * GRID_STEP[2])]])

sig_grid = 1.3 * GRID_STEP[1]
sig_grid2 = sig_grid ** 2
SIG_GRID = sig_grid2 * np.eye(2)
INV_SIG_GRID = np.mat(SIG_GRID).I
phi_x = np.zeros(NUM_STATE_FEATURES, 1)

#### action space
NUM_ACT = 3
ACT = np.mat([[0], [1], [2]])  # 0-push left, 1-no push, 2-push right
num_policy_param = NUM_STATE_FEATURES * NUM_ACT


def cal_score(theta, state):
    """
    Calculate score using RBN
    :return: a(action), scr(score)
    """
    y = state.y
    ## calculate feature values
    mu = np.zeros(NUM_ACT, 1)
    phi_x = np.zeros(NUM_STATE_FEATURES, 1)
    for th_ in range(NUM_STATE_FEATURES):
        temp = y - GRID_CENTERS[:, th_]
        phi_x[th_] = math.exp(-0.5 * temp.transpose() * INV_SIG_GRID * temp)
    for th_ in range(NUM_ACT):
        if th_ == 0:
            phi_xa = np.mat([phi_x, np.zeros(NUM_STATE_FEATURES, 1)])
        elif th_ == 1:
            phi_xa = np.mat([np.zeros(NUM_STATE_FEATURES, 1), phi_x, np.zeros(NUM_STATE_FEATURES, 1)])
        else:
            phi_xa = np.mat([np.zeros(NUM_STATE_FEATURES, 1), np.zeros(NUM_STATE_FEATURES, 1), phi_x])
        mu[th_] = math.exp(phi_xa.transpose() * theta)
    ## sample from probability
    temp = random.uniform(0, 1)
    if temp <= mu[0]:
        a = ACT[0]
        scr = np.mat([phi_x * (1 - mu[0]), -phi_x * mu[1], -phi_x * mu[2]])
    elif temp <= mu[1]:
        a = ACT[1]
        scr = np.mat([-phi_x * mu[0], phi_x * (1 - mu[1]), -phi_x * mu[2]])
    else:
        a = ACT[1]
        scr = np.mat([-phi_x * mu[0], -phi_x * mu[0], phi_x * (1 - mu[2])])
    return a, scr
