import car_env as car_env
import plot_funcion as my_plot
from RL_brain_DeepQ import DeepQNetwork
import numpy as np

# SIM_END_DISTANCE = car_env.ROAD_LENGTH - 200  # 在到达路的终点之前结束仿真
MAX_EPISODE = 70
time_tag = 0.0
total_steps = 0
TEST_CAR = 2
RL = DeepQNetwork(n_actions=len(car_env.ACTION_SPACE),
                  n_features=TEST_CAR * 2,
                  learning_rate=0.1, e_greedy=0.99,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001, )

if __name__ == '__main__':
    Carlist = []  # 车辆的数组
    # car1 = car_env.car(
    #     id=0,
    #     role='leader',
    #     ingaged_in_platoon=False,
    #     tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
    #     tar_speed=60.0 / 3.6,
    #     location=[0, 50]
    # )
    # car2 = car_env.car(
    #     id=1,
    #     role='follower',
    #     ingaged_in_platoon=False,
    #     tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
    #     tar_speed=60.0 / 3.6,
    #     location=[0, 25]
    # )
    # car3 = car_env.car(
    #     id=2,
    #     role='follower',
    #     ingaged_in_platoon=False,
    #     tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
    #     tar_speed=60.0 / 3.6,
    #     location=[0, 50]
    # )

    # 开始学习
    for i_episode in range(MAX_EPISODE):

        # 每个episode都要reset一下
        Carlist.clear()
        time_tag = 0.0
        car1 = car_env.car(
            id=0,
            role='leader',
            ingaged_in_platoon=False,
            tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
            tar_speed=60.0 / 3.6,
            location=[0, 50]
        )
        car2 = car_env.car(
            id=1,
            role='follower',
            ingaged_in_platoon=False,
            tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
            tar_speed=60.0 / 3.6,
            location=[0, 25]
        )
        # 将新车加入车队
        if len(Carlist) == 0:
            Carlist.append(car1)
            Carlist.append(car2)
        # 设置参与车队的车辆，根据build_platoon，更新是否加入platoon的标志位
        car_env.CarList_update_platoon_info(Carlist, des_platoon_size=2, build_platoon=True)
        observation = car_env.reset(Carlist)
        ep_r = 0

        # 开始每个episode的运算
        while True:
            # 时间戳更新
            time_tag += car_env.AI_DT

            action = RL.choose_action(observation)

            # 更新运动学参数。由于c++程序的3D和CarAI的时钟不同步，需要模仿那个程序进行多轮次更新
            # 实际上多轮次更新得到的数据更贴近连续值
            observation_, done, info = car_env.step_next(Carlist, time_tag, action)

            # 计算单步奖励
            reward = car_env.get_reward(observation_)

            # 存储
            RL.store_transition(observation, action, reward, observation_)

            ep_r += reward
            if total_steps > 1000:
                RL.learn()

            # 终止条件判断
            # print('time_tag:%.2f' % time_tag, ',now_car1_y:%.2f' % car1.location[1])
            if done:
                print('episode: ', i_episode,
                      ',ep_r: ', round(ep_r, 2),
                      ',epsilon: ', round(RL.epsilon, 2),
                      ',', info)
                break

            # 更新
            observation = observation_
            total_steps += 1

        # 画一下最后一个episode的结果
        if i_episode == MAX_EPISODE - 1:
            my_plot.plot_data(Carlist)

    # 绘制RL结果
    print(total_steps)
    RL.plot_cost()
