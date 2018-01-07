import math
import random
import numpy as np
import Param_class as param_class

#### state space
POS_RANGE = np.array([-1.2, 0.5])
VEL_RANGE = np.array([-0.07, 0.07])
GOAL = 0.5  # mountain-car中flag所在的位置
POS_MAP_RANGE = np.array([0, 1],dtype=float)
VEL_MAP_RANGE = np.array([0, 1],dtype=float)
GRID_SIZE = np.array([4, 4])
NUM_STATE_FEATURES = GRID_SIZE[0] * GRID_SIZE[1]

c_map_pos = np.mat([[POS_RANGE[0], 1], [POS_RANGE[1], 1]],dtype=float)
c_map_pos = c_map_pos.I * np.mat(POS_MAP_RANGE).transpose()
c_map_vel = np.mat([[VEL_RANGE[0], 1], [VEL_RANGE[1], 1]],dtype=float)
c_map_vel = c_map_vel.I * np.mat(VEL_MAP_RANGE).transpose()
c_map_pos = c_map_pos.getA1()
c_map_vel = c_map_vel.getA1()
GRID_STEP = np.array(
    [(POS_MAP_RANGE[1] - POS_MAP_RANGE[0]) / GRID_SIZE[0], (VEL_MAP_RANGE[1] - VEL_MAP_RANGE[0]) / GRID_SIZE[1]])
## fill in the center of RBF
GRID_CENTERS = np.zeros((2, NUM_STATE_FEATURES))
for i in range(GRID_SIZE[0]):
    for j in range(GRID_SIZE[1]):
        GRID_CENTERS[:, (i * GRID_SIZE[1]) + j] = np.mat(
            [POS_MAP_RANGE[0] + ((i + 0.5) * GRID_STEP[0]), VEL_MAP_RANGE[0] + ((j + 0.5) * GRID_STEP[1])])

sig_grid = 1.3 * GRID_STEP[0]
sig_grid2 = sig_grid ** 2
SIG_GRID = sig_grid2 * np.eye(2)
INV_SIG_GRID = np.mat(SIG_GRID).I
phi_x = np.zeros((NUM_STATE_FEATURES, 1))

#### action space
NUM_ACT = 2
ACT = np.array([-1, 1])  # -1-push left, 1-push right
num_policy_param = NUM_STATE_FEATURES * NUM_ACT


def random_reset():
    """
    reset state for simulation. 将state类实例化
    """
    rnd1 = random.uniform(0, 1)
    rnd2 = random.uniform(0, 1)
    # x = np.array(
    #     [((-0.6) - (-0.4)) * rnd1 + (-0.4), 0])  # set according to https://github.com/openai/gym/wiki/MountainCar-v0
    x = np.array(
        [(POS_RANGE[1] - POS_RANGE[0]) * rnd1 + POS_RANGE[0], (VEL_RANGE[1] - VEL_RANGE[0]) * rnd2 + VEL_RANGE[0]])
    y = np.array([c_map_pos[0] * x[0] + c_map_pos[1], c_map_vel[0] * x[1] + c_map_vel[1]])
    isgoal = 0
    state = param_class.state(x=x, y=y, isgoal=isgoal)
    return state


def cal_score(theta, state):
    """
    Calculate score using RBN
    """
    y = state.y
    ## calculate feature values
    mu = np.zeros((NUM_ACT, 1))
    phi_x = np.zeros((NUM_STATE_FEATURES, 1))
    for th_ in range(NUM_STATE_FEATURES):
        temp = np.mat(y - GRID_CENTERS[:, th_])
        phi_x[th_] = math.exp(-0.5 * np.dot(np.dot(temp, INV_SIG_GRID), temp.transpose()))
    for th_ in range(NUM_ACT):  # 0-push left, 1-no push, 2-push right
        zero_mat = np.zeros((NUM_STATE_FEATURES, 1))
        if th_ == 0:
            phi_xa = np.vstack((phi_x, zero_mat))
        else:
            phi_xa = np.vstack((zero_mat, phi_x))
        mu[th_] = math.exp(np.dot(phi_xa.transpose(), theta))
    mu = mu / mu.sum(axis=0)  # 对PG概率归一化
    ## sample from probability
    temp = random.uniform(0, 1)
    if temp <= mu[0]:
        a = ACT[0]
        scr = np.vstack((phi_x * (1 - mu[0]), -phi_x * mu[1]))
    else:
        a = ACT[1]
        scr = np.vstack((-phi_x * mu[0], phi_x * (1 - mu[1])))
    ## return
    return a, scr, mu


def cal_reward(state):
    # reward = np.abs(state.x[1]-(-0.5))  # 距离开始的地方越远，奖励越多
    reward = state.isgoal - 1
    return reward


def judge_done(state, done):
    if state.x[0] >= GOAL:
        state.isgoal = 1
        done = True
    else:
        state.isgoal = 0
        done = False


def convert_obs_to_state(state, observation, done):
    state.x[0] = observation[0]
    state.x[1] = observation[1]
    state.y = np.array([c_map_pos[0] * observation[0] + c_map_pos[1], c_map_vel[0] * observation[1] + c_map_vel[1]])
    if done:
        state.isgoal = 1
    else:
        state.isgoal = 0
