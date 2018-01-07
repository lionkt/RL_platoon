import math
import numpy as np
import Param_class as param_class
import Domain_func as domain


def step_next(state, a_old):
    """
    mountain car dynamics environment
    :param state:
    :param a_old:
    """
    x_old = state.x
    x = np.array([0.0, 0.0],dtype=float)
    temp = x_old[1] + 0.001 * a_old - 0.0025 * math.cos(3 * x_old[0])
    x[1] = max(domain.VEL_RANGE[0], min(temp, domain.VEL_RANGE[1]))  # update velocity
    temp = x_old[0] + x[1]
    x[0] = max(domain.POS_RANGE[0], min(temp, domain.POS_RANGE[1]))  # update position

    if x[0] <= domain.POS_RANGE[0]:
        x[1] = 0
    if x[0] >= domain.GOAL:
        x[0] = domain.GOAL
        x[1] = 0
    y = np.array([domain.c_map_pos[0] * x[0] + domain.c_map_pos[1], domain.c_map_vel[0] * x[1] + domain.c_map_vel[1]])
    n_state = state
    state.x =x
    state.y =y
    state.isgoal = 0
