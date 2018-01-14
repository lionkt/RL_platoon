import os

import gym

from Mountain_car_RL.DeepRL_method import DeepQNetwork
import Mountain_car_RL.mountain_car_env as mountain_car_env

# env = gym.make('MountainCar-v0')
# env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

MAX_train_episode = 100

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RL = DeepQNetwork(n_actions=2, n_features=2, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0001)

total_steps = 0

for i_episode in range(MAX_train_episode):
    if (i_episode + 1) % 20 == 0:
        print('=== Now finish', str((i_episode+1) / MAX_train_episode * 100), '% of ', str(MAX_train_episode), 'eps')

    observation = mountain_car_env.random_reset()
    ep_r = 0
    ep_step = 0
    while True:
        # env.render()

        action = RL.choose_action(observation)

        # observation_, reward, done, info = env.step(action)
        observation_, done = mountain_car_env.step_next(observation, action)

        # 车开得越高 reward 越大
        reward = mountain_car_env.cal_reward(observation_)

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        ep_step += 1
        if done:
            get = '| Get' if observation_[0] >= mountain_car_env.GOAL else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2),
                  '| Ep_step: ', str(ep_step))
            break

        if ep_step > 300:
            break

        observation = observation_
        total_steps += 1

        # RL.plot_cost()
