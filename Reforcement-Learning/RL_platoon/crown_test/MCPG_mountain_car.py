import gym
import numpy as np
import Param_class as parm_class
import Domain_func as domain

# from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
#                   replace_target_iter=300, memory_size=3000,
#                   e_greedy_increment=0.0001,)

total_steps = 0


def eval_performance(theta, learning_param):
    """
    Evaluate RL controller performance
    """
    step_avg = 0
    for eval_th in range(learning_param.num_episode_eval):
        t = 0
        done = False
        state = domain.random_reset()
        a, scr = domain.cal_score(theta=theta, state=state)
        while not done and t < learning_param.episode_len_max:
            observation_, reward, done, info = env.step(a)
            domain.convert_obs_to_state(state=state, observation=observation_, done=done)  # python的参数传递的是引用
            a, scr = domain.cal_score(theta=theta, state=state)
            t += 1
        # sum among different episodes
        step_avg = step_avg + t
    performance_avg = step_avg / learning_param.num_episode_eval
    return performance_avg


def MCPG(learning_param):
    """
    MC policy gradient method for mountain-car test
    """
    num_output = round(learning_param.num_update_max / learning_param.sample_interval + 1)
    theta = np.zeros((domain.num_policy_param, 1))
    performance_list = np.zeros((learning_param.num_trial, num_output))
    for trail_th in range(learning_param.num_trial):
        # TODO: 打印测试的结果
        alpha = learning_param.alpha_init
        for update_th in range(learning_param.num_update_max):
            if update_th % learning_param.sample_interval == 0:
                print('==== Trail:' + str(trail_th) + ', update:' + str(update_th) + ' ====')
            delta = np.zeros((domain.num_policy_param, 1))
            T = 0
            if update_th % learning_param.sample_interval == 0:  # 开始评估
                # TODO: 记录测试的时间
                eval_point = round(update_th / learning_param.sample_interval) + 1
                performance_list[trail_th, eval_point] = eval_performance(theta,
                                                                          learning_param)  # evaluate average steps
                # TODO: 打印测评估结果
                print('------ average step: ', str(performance_list[trail_th, eval_point]))
            ###### update PG parameters after M episodes
            for ep_th in range(learning_param.num_episode):
                t = 0
                score_path_sum = np.zeros((domain.num_policy_param, 1))
                state = domain.random_reset()
                done = False
                a, scr = domain.cal_score(theta, state)
                while not done and t < learning_param.episode_len_max:
                    observation_, reward, done, info = env.step(a)
                    domain.convert_obs_to_state(state=state, observation=observation_, done=done)  # python的参数传递的是引用
                    a, scr = domain.cal_score(theta=theta, state=state)
                    score_path_sum = score_path_sum + scr
                    delta = delta + reward * score_path_sum  # update delta for gradient calculation
                    t += 1
                T = T + t
            ##### update gradient for MCPG
            grad_MCPG = delta
            theta = theta + alpha * grad_MCPG
    # return parameters and
    return performance_list, theta


if __name__ == '__main__':
    learning_params = parm_class.learning_param(num_update_max=500, sample_interval=10, num_trial=1, gamma=0.95,
                                                num_episode=5, episode_len_max=200, num_episode_eval=10,
                                                alpha_init=0.025)
    # begin train and evaluation
    MCPG(learning_param=learning_params)


# for i_episode in range(10):
#
#     observation = env.reset()
#     ep_r = 0
#     while True:
#         env.render()
#
#         action = RL.choose_action(observation)
#
#         observation_, reward, done, info = env.step(action)
#
#         position, velocity = observation_
#
#         # 车开得越高 reward 越大
#         reward = abs(position - (-0.5))
#
#         RL.store_transition(observation, action, reward, observation_)
#
#         if total_steps > 1000:
#             RL.learn()
#
#         ep_r += reward
#         if done:
#             get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
#             print('Epi: ', i_episode, get, '| Ep_r: ', round(ep_r, 4), '| Epsilon: ', round(RL.epsilon, 2))
#             break
#
#         observation = observation_
#         total_steps += 1
#
# RL.plot_cost()
