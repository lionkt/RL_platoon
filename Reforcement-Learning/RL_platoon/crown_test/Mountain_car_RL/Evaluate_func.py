import os

import gym

from Mountain_car_RL.DeepRL_method import DeepQNetwork
import Mountain_car_RL.mountain_car_env as mountain_car_env


def eval_mountain_car(RL, eval_eps, reset_method=None, reward_function=None):
    total_steps = 0
    for ep_i in range(eval_eps):
        observation = mountain_car_env.random_reset(method=reset_method)
        ep_r = 0
        ep_step = 0

        # begin train
        while True:

            action = RL.choose_action(observation)

            # observation_, reward, done, info = env.step(action)
            observation_, done = mountain_car_env.step_next(observation, action)

            # 车开得越高 reward 越大
            reward = mountain_car_env.cal_reward(observation_, reward_function)

            ep_r += reward
            ep_step += 1

            if ep_step > 300:
                break
            if done:
                break

            observation = observation_
        # begin eval
        total_steps += ep_step

    avg_steps = round(total_steps / eval_eps)
    return avg_steps
