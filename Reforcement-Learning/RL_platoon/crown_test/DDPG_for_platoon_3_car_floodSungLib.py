from floodsungLib.ddpg import *
import numpy as np
import car_env_DDPG_3cars as car_env
import plot_funcion as my_plot
import plot_train as train_plot
import gc
import os
gc.enable()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set random seed
np.random.seed(1)
tf.set_random_seed(1)


# hyper params
MAX_EPISODES = 10 # 3000
TEST = 10


LOAD_NN = False
# LOAD_NN = True
OUTPUT_GRAPH = True
# USE_RL_METHOD = False    # 判断是用传统的跟驰控制，还是用RL控制
USE_RL_METHOD = True    # 判断是用传统的跟驰控制，还是用RL控制
INIT_CAR_DISTANCE = 25  # 初始时车辆的间隔


STATE_DIM = car_env.STATE_DIM
ACTION_DIM = car_env.ACTION_DIM
ACTION_BOUND = car_env.ACTION_BOUND


######## build DDPG networks ########
agent = DDPG(STATE_DIM, ACTION_DIM, ACTION_BOUND, LOAD_NN, OUTPUT_GRAPH)




def train():
    # record field
    reward_list = []
    explore_list = []
    info_list = []
    observation_list = []
    plot_interval = 8  # 绘制训练图像的次数
    plot_iter = 1  # 当前的训练图绘制次数
    # train params
    var = 5  # control exploration, original 2.5
    var_damp = 0.99996  # var damping ratio, original 0.99995
    last_a = 0  # 上一个加速度值
    Carlist = []
    for ep in range(MAX_EPISODES):
        # 每个episode都要reset一下
        Carlist.clear()
        time_tag = 0.0
        car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6,
                           location=[0, INIT_CAR_DISTANCE * 2])
        car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6,
                           location=[0, INIT_CAR_DISTANCE])
        car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 0])
        # 将新车加入车队
        if len(Carlist) == 0:
            Carlist.append(car1)
            Carlist.append(car2)
            Carlist.append(car3)
        # 设置参与车队的车辆，根据build_platoon，更新是否加入platoon的标志位
        CarList_update_platoon_info(Carlist, des_platoon_size=len(Carlist), build_platoon=True)
        s = car_env.reset(Carlist)
        ep_reward = 0

        while True:
            # 时间戳更新
            time_tag += car_env.AI_DT

            # Added exploration noise
            # a = agent.OU_noise_action(s)
            a = agent.normal_noise_action(s, var)
            a = np.clip(a, ACTION_BOUND[0], ACTION_BOUND[1])
            s_, done, info = car_env.step_next(Carlist, time_tag, action=a)
            r = car_env.get_reward_function(s_, (Carlist[2].acc - last_a) / car_env.AI_DT)
            # r = car_env.get_reward_table(s_)

            # 旧加速度更新
            last_a = Carlist[2].acc

            # 存储当前链，并进行学习
            var = agent.perceive(s, a, r, s_, done, var, var_damp)

            # 更新状态
            s = s_
            ep_reward += r

            if done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep, result, '| R: %i' % int(ep_reward), '| Explore: %.2f' % var, '| info: ', info, '| dist-err(f1-f2):%.2f' % s[1],
                      '| speed-err(f1-f2):%.2f' % s[0], '| speed-err(le-f2):%.2f' % s[2])
                ## save data for plot
                reward_list.append(int(ep_reward))
                explore_list.append(var)
                info_list.append(info)
                observation_list.append(s)
                break

        # 画一下最后一次的图像
        if ep == MAX_EPISODES - 1:
            train_plot.plot_train_core(reward_list, explore_list, info_list, observation_list, write_flag=False,
                                       title_in=1 * 100)
            my_plot.plot_data(Carlist, write_flag=True)
        # 画一下训练过程中的图像
        if ep == MAX_EPISODES // plot_interval * plot_iter:
            plot_iter += 1
            train_plot.plot_train_core(reward_list, explore_list, info_list, observation_list, write_flag=False,
                                       title_in=ep / MAX_EPISODES * 100)

    # save nn params and graph
    agent.save()


def eval():
    Carlist = []
    # 每个episode都要reset一下
    Carlist.clear()
    time_tag = 0.0
    car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, INIT_CAR_DISTANCE * 3])
    car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, INIT_CAR_DISTANCE * 2])
    car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, INIT_CAR_DISTANCE])
    car4 = car_env.car(id=3, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 0])
    car5 = car_env.car(id=4, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, -INIT_CAR_DISTANCE])
    # 将新车加入车队
    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        Carlist.append(car3)
        Carlist.append(car4)  # Carlist.append(car5)
    CarList_update_platoon_info(Carlist, des_platoon_size=len(Carlist), build_platoon=True)  # 把车辆加入车队
    s = car_env.reset(Carlist)
    done = False
    # main loop for eval
    while True:
        time_tag += car_env.AI_DT  # 时间戳更新
        # 多车同时加入仿真的计算
        Carlist[0].calculate(Carlist[0], STRATEGY='ACC', time_tag=time_tag, action=None)  # 先算头车
        Carlist[1].calculate(Carlist[0:2], STRATEGY='ACC', time_tag=time_tag, action=None)  # 先算第二辆
        for car_index in range(len(Carlist)):
            if car_index <= 1:
                continue
            if car_index == 2:
                temp_list = []  # 3辆车的数组
                temp_list.append(Carlist[car_index - 2])
                temp_list.append(Carlist[car_index - 1])
                temp_list.append(Carlist[car_index])
            elif car_index >= 3:
                temp_list = []  # 3辆车的数组
                temp_list.append(Carlist[0])
                temp_list.append(Carlist[car_index - 1])
                temp_list.append(Carlist[car_index])
            s, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 先读取一下当前的状态
            a = agent.action(s)  # 根据当前状态，从训练好的网络选择动作
            temp_list[2].calculate(temp_list, STRATEGY='RL', time_tag=time_tag, action=a)  # 将输入的动作用于运算
            s_, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 更新一下当前的状态

        # 信息更新
        turns = 0
        while turns <= car_env.AI_DT:
            car_env.CarList_update_info_core(Carlist, car_env.UPDATE_TIME_PER_DIDA)
            turns += car_env.UPDATE_TIME_PER_DIDA

        # 判断仿真是否结束
        if done:
            break

    my_plot.plot_data(Carlist, write_flag=True)


# 根据build_platoon，更新是否加入platoon的信息
def CarList_update_platoon_info(Carlist, des_platoon_size, build_platoon):
    if build_platoon == False:
        for single_car in Carlist:
            single_car.engaged_in_platoon = False
    else:
        for single_car in Carlist:
            single_car.leader = Carlist[0]
        assert len(Carlist) >= des_platoon_size, '期望长度大于CarList总长度'

        for single_car in Carlist:
            single_car.engaged_in_platoon = True


if __name__ == '__main__':
    if LOAD_NN == False:
        train()
    else:
        eval()