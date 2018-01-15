"""
Policy Gradient, Reinforcement Learning.

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
import os
from Mountain_car_RL.DeepRL_method import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import Mountain_car_RL.mountain_car_env as mountain_car_env
import Mountain_car_RL.Evaluate_func as eval_module


DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502


MAX_train_episode = 1000
MAX_episode_length = 500
Eval_interval = 50
Eval_episode = 100

np.random.seed(2)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# env = gym.make('MountainCar-v0')
# env.seed(1)  # reproducible, general Policy gradient has high variance
# env = env.unwrapped
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=mountain_car_env.NUM_ACT,
    n_features=mountain_car_env.NUM_FEATURE,
    learning_rate=0.002,
    reward_decay=0.995,
    # output_graph=True,
)


if __name__ == '__main__':

    avg_steps_list = []

    for i_episode in range(MAX_train_episode):
        if (i_episode + 1) % Eval_interval == 0:
            print('=== Now finish %.2f' %((i_episode + 1) / MAX_train_episode * 100), '% of ', str(MAX_train_episode), 'eps')

        # begin eval
        if (i_episode + 1) % Eval_interval == 0 or i_episode == 0:
            avg_steps = eval_module.eval_mountain_car(RL=RL, eval_eps=Eval_episode, reset_method=3, reward_function='original')
            avg_steps_list.append(avg_steps)
            print('--- eval, avg steps: %.3f' %(avg_steps))

        # observation = env.reset()
        observation = mountain_car_env.random_reset(method=3)
        ep_step = 0

        while True:

            # action = RL.choose_action(observation)
            # observation_, reward, done, info = env.step(action)  # reward = -1 in all cases
            # RL.store_transition(observation, action, reward)

            action = RL.choose_action(observation)
            observation_, done = mountain_car_env.step_next(observation, action)
            reward = mountain_car_env.cal_reward(observation_, reward_function='original')
            RL.store_transition(observation, action, reward)

            ep_step += 1

            if done:
                # calculate running reward
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                vt = RL.learn()  # train

                # print('Epi: ', i_episode,
                #       '| Ep_r: ', round(running_reward, 4),
                #       '| Ep_step: ', str(ep_step))
                break

            if ep_step >= MAX_episode_length:
                break

            observation = observation_

    root_path = '../OutputImg/Mountain_car/'
    output_file_name = 'PG' + '_MaxEp=' + str(MAX_train_episode) + '_MaxEpLen=' + str(MAX_episode_length) + '_AvgSteps.txt'
    write_buffer = np.array(avg_steps_list).transpose()
    np.savetxt(root_path + output_file_name, write_buffer)