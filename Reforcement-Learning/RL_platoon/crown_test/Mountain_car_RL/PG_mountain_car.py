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

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502


MAX_train_episode = 500

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

env = gym.make('MountainCar-v0')
env.seed(1)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

for i_episode in range(MAX_train_episode):

    observation = env.reset()
    ep_step = 0

    while True:

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)  # reward = -1 in all cases

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

            print('Epi: ', i_episode,
                  '| Ep_r: ', round(running_reward, 4),
                  '| Epsilon: ', round(RL.epsilon, 2),
                  '| Ep_step: ', str(ep_step))

            break

        if ep_step >= 500:
            break

        observation = observation_
